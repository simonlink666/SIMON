#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import gc
import argparse
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageFilter
import cv2

import torch
import torch.nn.functional as F
from torchvision import transforms
import open_clip
from tqdm import tqdm


def _device_str_from_auto(dev: Optional[str]) -> str:
    s = str(dev) if dev is not None else "cuda:0"
    if isinstance(dev, int):
        return f"cuda:{dev}"
    if s.startswith(("cuda", "cpu", "mps")):
        return s
    return "cuda:0"

def _load_eeg_pt(path: str, avg: bool) -> Dict[str, torch.Tensor]:
    """Loads the EEG pt file to get the image list, ensuring alignment."""
    print(f"[INFO] Loading mapping from: {path}")
    data = torch.load(path, map_location='cpu', weights_only = False)
    out = {}
    out["img"] = data["img"][:, 0] if avg else data["img"].reshape(-1)
    return out

def _pil_to_tensor_normalized():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

@torch.no_grad()
def encode_images_openclip(vlmodel, imgs: List[Image.Image], preprocess, device: torch.device, normalize: bool = True) -> torch.Tensor:
    dev = next(vlmodel.parameters()).device
    # Ensure preprocess is not None
    if preprocess is None:
        preprocess = _pil_to_tensor_normalized()
    
    # Process batch
    T = torch.stack([preprocess(p) for p in imgs]).to(dev)
    feats = vlmodel.encode_image(T)
    if normalize: 
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.float().cpu()


_GLOBAL_BIREFNET = None
_GLOBAL_BIREFNET_TFM = None
_GLOBAL_BIREFNET_DEV = None
_GLOBAL_SUM = None
_GLOBAL_SUM_DEV = None

def _centroid_of_bool_mask(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        h, w = mask.shape
        return w * 0.5, h * 0.5
    return float(xs.mean()), float(ys.mean())

def _sum_peak_center_by_centroid(S: np.ndarray, fg_mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    S = S.astype(np.float32)
    maxv = float(S.max())
    mask_max = (S == maxv)
    if fg_mask is not None:
        fg_mask = fg_mask.astype(bool)
        masked = mask_max & fg_mask
        if masked.any():
            return _centroid_of_bool_mask(masked)
    return _centroid_of_bool_mask(mask_max)

def _project_to_nearest_mask_point(seed_xy: Tuple[float, float], ys: np.ndarray, xs: np.ndarray) -> Tuple[float, float]:
    if xs.size == 0:
        return seed_xy
    x0, y0 = float(seed_xy[0]), float(seed_xy[1])
    dx = xs.astype(np.float32) - x0
    dy = ys.astype(np.float32) - y0
    d2 = dx * dx + dy * dy
    j = int(d2.argmin())
    return float(xs[j]), float(ys[j])

def _fps_in_mask_from_seed(xs: np.ndarray, ys: np.ndarray, sal_vals: Optional[np.ndarray],
                           seed_xy: Tuple[float, float], k: int, alpha_gamma: float = 0.5) -> List[Tuple[float, float]]:
    if k <= 0: return []
    if xs.size == 0: return [seed_xy] * k

    seed_xy = _project_to_nearest_mask_point(seed_xy, ys, xs)
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    seed = np.array([seed_xy[0], seed_xy[1]], dtype=np.float32)[None, :]
    dists = np.linalg.norm(pts - seed, axis=1)

    if sal_vals is not None:
        w = np.power(sal_vals.astype(np.float32) + 1e-6, float(alpha_gamma))
    else:
        w = None

    centers = [(float(seed_xy[0]), float(seed_xy[1]))]
    if k == 1: return centers

    for _ in range(1, k):
        scores = dists if w is None else (dists * w)
        idx = int(scores.argmax())
        cx_i, cy_i = float(pts[idx, 0]), float(pts[idx, 1])
        centers.append((cx_i, cy_i))
        new_d = np.linalg.norm(pts - np.array([cx_i, cy_i], dtype=np.float32)[None, :], axis=1)
        dists = np.minimum(dists, new_d)
    return centers

def _init_birefnet(model_id: str, device: torch.device, image_size: int = 1024, fp16: bool = False):
    from transformers import AutoModelForImageSegmentation
    print(f"[INFO] Initializing BiRefNet: {model_id}")
    model = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
    model.eval().to(device)
    if fp16 and device.type == "cuda":
        model.half()
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, tfm

@torch.no_grad()
def _get_alpha_birefnet(pil_img: Image.Image) -> np.ndarray:
    global _GLOBAL_BIREFNET, _GLOBAL_BIREFNET_TFM, _GLOBAL_BIREFNET_DEV
    img_rgb = pil_img.convert("RGB")
    x = _GLOBAL_BIREFNET_TFM(img_rgb).unsqueeze(0).to(_GLOBAL_BIREFNET_DEV)
    out = _GLOBAL_BIREFNET(x)

    pred = None
    if torch.is_tensor(out): pred = out
    elif isinstance(out, (list, tuple)):
        for t in reversed(out):
            if torch.is_tensor(t): pred = t; break
    elif isinstance(out, dict):
        for _, v in out.items():
            if torch.is_tensor(v): pred = v

    if pred.ndim == 3: pred = pred.unsqueeze(1)
    elif pred.ndim == 4 and pred.shape[1] != 1: pred = pred[:, :1, :, :]

    pred32 = pred.float()
    flat = pred32.flatten()
    in01 = ((flat >= 0.0) & (flat <= 1.0)).float().mean().item()
    prob = torch.sigmoid(pred32) if in01 < 0.995 else pred32

    prob = F.interpolate(prob, size=(pil_img.size[1], pil_img.size[0]), mode="bilinear", align_corners=False)
    alpha = prob[0, 0].detach().cpu().numpy().astype(np.float32)
    return np.clip(alpha, 0.0, 1.0)

def _init_sum_model(ckpt_path: str, device: torch.device):
    print(f"[INFO] Initializing SUM: {ckpt_path}")
    import sys
    sum_path = os.path.join(os.path.dirname(__file__), 'SUM')
    if sum_path not in sys.path:
        sys.path.append(sum_path)
    
    from net import SUM
    from net.configs.config_setting import setting_config

    config = setting_config
    model_cfg = config.model_config

    model = SUM(
        num_classes=model_cfg["num_classes"],
        input_channels=model_cfg["input_channels"],
        depths=model_cfg["depths"],
        depths_decoder=model_cfg["depths_decoder"],
        drop_path_rate=model_cfg["drop_path_rate"],
    )
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model

@torch.no_grad()
def _get_saliency_sum(pil_img: Image.Image, condition: int, input_size: int, smooth_sigma: float) -> np.ndarray:
    global _GLOBAL_SUM, _GLOBAL_SUM_DEV
    img_rgb = pil_img.convert("RGB")
    w0, h0 = img_rgb.size
    tfm = transforms.Compose([
        transforms.Resize((int(input_size), int(input_size))),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tfm(img_rgb).unsqueeze(0).to(_GLOBAL_SUM_DEV)
    one_hot = torch.zeros((1, 4), device=_GLOBAL_SUM_DEV)
    one_hot[0, int(condition)] = 1

    out = _GLOBAL_SUM(x, one_hot)
    t = out[-1] if isinstance(out, (list, tuple)) else out
    
    if t.ndim == 3: t = t.unsqueeze(1)
    elif t.ndim == 4 and t.shape[1] != 1: t = t[:, :1, :, :]
    
    t = t.float()
    flat = t.flatten()
    if ((flat >= 0.0) & (flat <= 1.0)).float().mean().item() < 0.995:
        t = torch.sigmoid(t)

    t = F.interpolate(t, size=(h0, w0), mode="bilinear", align_corners=False)
    S = t[0, 0].detach().cpu().numpy().astype(np.float32)
    S = np.clip(S, 0.0, 1.0)

    if smooth_sigma > 0:
        S_img = Image.fromarray((S * 255).astype(np.uint8))
        S_img = S_img.filter(ImageFilter.GaussianBlur(radius=float(smooth_sigma)))
        S = np.array(S_img, dtype=np.float32) / 255.0
    return S

def _blend_pyramid_by_sigma(orig: np.ndarray, sigma_map: np.ndarray, min_sigma: float, max_sigma: float, levels: int) -> np.ndarray:
    L = max(int(levels), 2)
    sigmas = np.linspace(min_sigma, max_sigma, L, dtype=np.float32)
    step = (max_sigma - min_sigma) / (L - 1) if L > 1 else 1.0
    pil = Image.fromarray(orig)
    blurred = []
    for s in sigmas:
        if s <= 1e-6: blurred.append(orig.astype(np.float32))
        else:
            b = np.array(pil.filter(ImageFilter.GaussianBlur(radius=float(s))), dtype=np.float32)
            blurred.append(b)
    blurred = np.stack(blurred, axis=0)
    W = np.maximum(0.0, 1.0 - np.abs(sigma_map[None, ...] - sigmas[:, None, None]) / max(step, 1e-6))
    W_sum = W.sum(axis=0, keepdims=True).clip(min=1e-6)
    W = W / W_sum
    comp = (W[..., None] * blurred).sum(axis=0)
    return np.clip(comp, 0, 255).astype(np.uint8)

def _soft_from_alpha(alpha01: np.ndarray, feather: float) -> np.ndarray:
    a_img = Image.fromarray(np.clip(alpha01 * 255.0, 0, 255).astype(np.uint8))
    soft = np.array(a_img.filter(ImageFilter.GaussianBlur(radius=float(feather))), dtype=np.float32) / 255.0
    return np.clip(soft, 0.0, 1.0)

def _foveated_bg_with_fgmask(orig: np.ndarray, fg_alpha: np.ndarray, min_sigma: float, max_sigma: float, levels: int, feather: float) -> np.ndarray:
    soft = _soft_from_alpha(fg_alpha, feather=feather)
    sigma_map = min_sigma * soft + max_sigma * (1.0 - soft)
    comp = _blend_pyramid_by_sigma(orig, sigma_map, min_sigma, max_sigma, levels)
    comp = comp * (1.0 - soft[..., None]) + orig.astype(np.float32) * soft[..., None]
    return np.clip(comp, 0, 255).astype(np.uint8)

def _radius_from_fg(alpha_fg: np.ndarray, h: int, w: int) -> float:
    mask = alpha_fg > 1e-3
    if mask.any():
        ys, xs = np.where(mask)
        R = 0.5 * np.hypot(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    else:
        R = 0.45 * np.hypot(h, w)
    return max(float(R), 1.0)

def _sample_centers_sum_seed_birefnet_fg(S: np.ndarray, alpha_fg: np.ndarray, k: int, fg_thr: float, downsample: int, alpha_gamma: float) -> List[Tuple[float, float]]:
    H, W = S.shape
    fg_mask = (alpha_fg > float(fg_thr))
    seed = _sum_peak_center_by_centroid(S, fg_mask=fg_mask)
    
    cand = fg_mask
    ys, xs = np.where(cand)
    if downsample > 1 and ys.size > 0:
        ys, xs = ys[::downsample], xs[::downsample]

    sal_vals = S[ys, xs].astype(np.float32) if ys.size > 0 else None
    centers = _fps_in_mask_from_seed(
        xs=xs, ys=ys, sal_vals=sal_vals, seed_xy=seed, k=int(k), alpha_gamma=float(alpha_gamma)
    )
    return [(float(np.clip(x, 0, W - 1)), float(np.clip(y, 0, H - 1))) for (x, y) in centers]

def _secondary_radial(orig: np.ndarray, alpha: np.ndarray, centers: List[Tuple[float, float]], 
                      min_sigma: float, max_sigma: float, levels: int, gamma: float) -> List[np.ndarray]:
    outs = []
    for (cx, cy) in centers:
        h, w, _ = orig.shape
        R = _radius_from_fg(alpha, h=h, w=w)
        ys, xs = np.arange(h), np.arange(w)
        X, Y = np.meshgrid(xs, ys)
        dist = np.hypot(X - cx, Y - cy)
        t = np.clip(dist / R, 0.0, 1.0) ** float(gamma)
        sigma_map = min_sigma + (max_sigma - min_sigma) * t
        outs.append(_blend_pyramid_by_sigma(orig, sigma_map, min_sigma, max_sigma, levels))
    return outs


def process_low_level_image(image_path: str, k_size: int = 31) -> Image.Image:
    """
    Reads an image, resizes to 224x224, applies Gaussian Blur using OpenCV,
    and returns a PIL Image suitable for OpenCLIP.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    blurred_np = cv2.GaussianBlur(image_np, (k_size, k_size), 0)
    return Image.fromarray(cv2.cvtColor(blurred_np, cv2.COLOR_BGR2RGB))

def main():
    p = argparse.ArgumentParser(description="Unified Extract Embedding (High & Low Level)")
    
    # Common Args
    p.add_argument("--mode", type=str, default="high", choices=["high", "low"], help="Extraction mode")
    p.add_argument("--eeg_pt", type=str, required=True, help="Input EEG/Image mapping .pt file")
    p.add_argument("--out_pt", type=str, required=True, help="Target path for output matrix")
    p.add_argument("--data_dir", type=str, required=True, help="Root directory for things-eeg data")
    p.add_argument("--gpu", type=str, default="0", help="GPU ID")
    p.add_argument("--avg", action="store_true", help="If using avg mode for loading EEG pt (High level only usually)")
    
    # Low Level Specific Args
    p.add_argument("--blur_k", type=int, default=31, help="Kernel size for Gaussian Blur (Low level mode)")
    
    # High Level Specific Config (Defaults matching original script)
    p.add_argument("--birefnet_id", type=str, default="ZhengPeng7/BiRefNet")
    p.add_argument("--sum_ckpt", type=str, default="SUM/net/pre_trained_weights/sum_model.pth")
    
    args = p.parse_args()

    # Environment Setup
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running Mode: {args.mode.upper()} on {device}")

    clip_model_name = "RN50"
    pretrained_tag = "openai" # Standard for RN50
    
    print(f"[INFO] Loading OpenCLIP {clip_model_name} ({pretrained_tag})...")
    # Forcing device to string format for open_clip compatibility if needed
    dev_str = _device_str_from_auto("cuda:0" if device.type == "cuda" else "cpu")
    vlmodel, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, device=dev_str, pretrained=pretrained_tag
    )
    vlmodel.eval()

    loaded_data = _load_eeg_pt(args.eeg_pt, avg=args.avg)
    
    
    unique_image_names = sorted(list(set(loaded_data["img"])))
    print(f"[INFO] Total unique images to process: {len(unique_image_names)}")

    img_root = os.path.join(args.data_dir, "Image_set_Resize")
    
    if args.mode == "high":
        global _GLOBAL_BIREFNET, _GLOBAL_BIREFNET_TFM, _GLOBAL_BIREFNET_DEV
        global _GLOBAL_SUM, _GLOBAL_SUM_DEV
        
        # SUM
        _GLOBAL_SUM_DEV = device
        _GLOBAL_SUM = _init_sum_model(args.sum_ckpt, device=device)
        
        # BiRefNet
        _GLOBAL_BIREFNET_DEV = device
        _GLOBAL_BIREFNET, _GLOBAL_BIREFNET_TFM = _init_birefnet(
            model_id=args.birefnet_id, device=device
        )
        cv2.setNumThreads(0)

    batch_size = 128 
    
    final_features_list = []

    clip_preprocess = preprocess if preprocess is not None else _pil_to_tensor_normalized()

    # Iterate in chunks
    for i in tqdm(range(0, len(unique_image_names), batch_size), desc=f"Processing {args.mode}"):
        batch_names = unique_image_names[i : i + batch_size]
        
        batch_pil_images = []
        
        if args.mode == "high":
            sum_cond = 1
            sum_input_size = 256
            sal_smooth = 0.0
            
            # Temporary storage for this batch's mean features
            batch_high_feats = []

            for name in batch_names:
                pth = os.path.join(img_root, name)
                if not os.path.exists(pth):
                    pass

                orig = Image.open(pth).convert("RGB").resize((224, 224), Image.BICUBIC)
                orig_np = np.array(orig, dtype=np.uint8)

                # Saliency
                map01 = _get_saliency_sum(orig, condition=sum_cond, input_size=sum_input_size, smooth_sigma=sal_smooth)
                
                # Segmentation
                alpha_fg = _get_alpha_birefnet(orig)

                # BG Blur 
                comp0 = _foveated_bg_with_fgmask(
                    orig_np, alpha_fg, min_sigma=0.0, max_sigma=0.5, levels=6, feather=4.0
                )

                # Centers
                centers = _sample_centers_sum_seed_birefnet_fg(
                    map01, alpha_fg, k=3, fg_thr=0.5, downsample=4, alpha_gamma=0.5
                )

                # Radial Variants
                variants = _secondary_radial(
                    comp0, alpha_fg, centers, min_sigma=0.0, max_sigma=2.0, levels=4, gamma=1.2
                )

                # Encode variants and mean
                var_pil = [Image.fromarray(v) for v in variants]
                feats_k = encode_images_openclip(vlmodel, var_pil, clip_preprocess, device, normalize=True)
                batch_high_feats.append(feats_k.mean(dim=0))

            # Stack batch results
            if len(batch_high_feats) > 0:
                final_features_list.append(torch.stack(batch_high_feats, dim=0))

        else:
            for name in batch_names:
                pth = os.path.join(img_root, name)
                if not os.path.exists(pth):
                    pass
                # Apply Low Level Blur
                processed_pil = process_low_level_image(pth, k_size=args.blur_k)
                batch_pil_images.append(processed_pil)
            
            # Encode entire batch at once
            if len(batch_pil_images) > 0:
                with torch.no_grad():
                    # Helper handles stacking and normalization
                    feats = encode_images_openclip(vlmodel, batch_pil_images, clip_preprocess, device, normalize=False)
                    final_features_list.append(feats)

    if len(final_features_list) > 0:
        F_all = torch.cat(final_features_list, dim=0)
        
        save_dir = os.path.dirname(args.out_pt)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        torch.save(F_all, args.out_pt)
        print(f"[SUCCESS] Saved {args.mode} embeddings to: {args.out_pt}")
        print(f"[INFO] Final Shape: {F_all.shape}")
    else:
        print("[WARN] No features extracted. Check paths.")

    # Cleanup
    del vlmodel
    if args.mode == "high":
        del _GLOBAL_SUM, _GLOBAL_BIREFNET
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()