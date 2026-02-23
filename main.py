import argparse
import os
import shutil
import json
import importlib
import numpy as np
import pandas as pd
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import AdamW, Adam, SGD 

import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf

# User defined modules
from base.utils import *
import base.lorentz as L

# torch.cuda.empty_cache()

def load_model(config, train_loader, test_loader):
    model = {}
    for k, v in config['models'].items():
        print(f"init {k}")
        model[k] = instantiate_from_config(v)

    pl_model = PLModel(model, config, train_loader, test_loader)
    return pl_model

class PLModel(pl.LightningModule):
    def __init__(self, model,config,train_loader,test_loader, model_type = 'RN50'):
        super().__init__()

        self.config = config
        for key, value in model.items():
            setattr(self, f"{key}", value)

        self.criterion = Hyperbolic_CL()

        self.all_predicted_classes = []
        self.all_true_labels = []

        self.z_dim = self.config['z_dim']
        print(self.z_dim)
        self.sim = np.ones(len(train_loader.dataset))

        self.mAP_total = 0
        self.match_similarities = []

        # ======= visual_alpha =======
        backbone_name = self.config.get('vision_backbone', 'RN50')
        
        if backbone_name.startswith('RN'): # e.g., RN50, RN101
            self.visual_alpha = 1.0
            print(f"[{backbone_name}] Detected ResNet-based backbone. Setting visual_alpha = 1.0")
        else:
            self.visual_alpha = (1.0 / self.z_dim) ** 0.5
            print(f"[{backbone_name}] Detected ViT/Other backbone. Setting visual_alpha = sqrt(1/{self.z_dim}) = {self.visual_alpha:.6f}")
    
  
    def forward(self, batch,sample_posterior=False):
 
        eeg = batch['eeg']
        high_level_feature  = batch['high_level_feature']
        low_level_feature = batch['low_level_feature']

        eeg_alpha = self.brain.eeg_alpha.exp()
        visual_alpha = self.visual_alpha
        curv = self.brain.curv.exp()

        eeg_z = self.brain(eeg)
        #calculate the interpolation coefficient
        score = self.brain.score(high_level_feature.float())

        #We find the that normalized the sementic features are more effective
        high_level_feature = high_level_feature / high_level_feature.norm(dim=-1, keepdim=True)

        # map to Lorentz space
        eeg_z = eeg_alpha*eeg_z
        eeg_z = L.exp_map0(eeg_z, curv)

        # apply linear layer
        high_level_feature = self.brain.image_projection1(high_level_feature.float())
        low_level_feature = self.brain.image_projection2(low_level_feature.float())

        # map to Lorentz space
        high_level_feature = visual_alpha*high_level_feature
        high_level_feature = L.exp_map0(high_level_feature, curv)
        low_level_feature = visual_alpha*low_level_feature
        low_level_feature = L.exp_map0(low_level_feature, curv)


        # interpolation following the geodesic
        interpolated_z = L.geodesic_interploation(high_level_feature, low_level_feature,score,curv)

        # apply contrastive learning
        loss = self.criterion(eeg_z, interpolated_z, curv)

        return eeg_z, interpolated_z, loss, curv

    

    def training_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss, curv = self(batch,sample_posterior=True)


        self.log('train_loss', loss, on_step=True, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        
        similarity = L.pairwise_dist(eeg_z, img_z, curv)
        top_kvalues, top_k_indices = similarity.topk(5,largest=False, dim=-1)


        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        if batch_idx == self.trainer.num_training_batches - 1:
            all_predicted_classes = np.concatenate(self.all_predicted_classes,axis=0)
            all_true_labels = np.array(self.all_true_labels)
            top_1_predictions = all_predicted_classes[:, 0]
            top_1_correct = top_1_predictions == all_true_labels
            top_1_accuracy = sum(top_1_correct)/len(top_1_correct)
            top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
            top_k_accuracy = sum(top_k_correct)/len(top_k_correct)
            self.log('train_top1_acc', top_1_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
            self.log('train_top5_acc', top_k_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
            self.all_predicted_classes = []
            self.all_true_labels = []

        return loss


    def validation_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss, curv = self(batch,sample_posterior=True)

        B, D = eeg_z.shape
        label = torch.arange(0, batch_size).to(self.device)

        similarity = L.pairwise_dist(eeg_z, img_z,curv)
        top_kvalues, top_k_indices = similarity.topk(5,largest=False, dim=-1)

        self.all_predicted_classes.append(top_k_indices.cpu().numpy())

        self.all_true_labels.extend(label.cpu().numpy())

        self.log('val_loss', loss, on_step=False, on_epoch=True,
                prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        
        return loss
    

    def on_validation_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes,axis=0)
        all_true_labels = np.array(self.all_true_labels)
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct)/len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct)/len(top_k_correct)
        self.log('val_top1_acc', top_1_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
        self.log('val_top5_acc', top_k_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
        self.all_predicted_classes = []
        self.all_true_labels = []


    def test_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss, curv = self(batch,sample_posterior=True)
      
        high_level = batch['high_level_feature']
        score = self.brain.score(high_level.float())
        self.score = score.mean()


        self.log('test_loss', loss, on_step=False, on_epoch=True,
                prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)

        label = torch.arange(0, batch_size).to(self.device)

        similarity = L.pairwise_dist(eeg_z, img_z,curv)
        top_kvalues, top_k_indices = similarity.topk(5,largest=False, dim=-1)

        mAP = 0.0

        # label = batch['label']
        self.all_true_labels.extend(label.cpu().numpy())
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        self.match_similarities.extend(similarity.diag().detach().cpu().tolist())
        self.mAP_total += mAP

        for i in range(similarity.shape[0]):
            true_index = i
            sims = similarity[i, :]
            sorted_indices = torch.argsort(sims)
            rank = (sorted_indices == true_index).nonzero()[0][0] + 1
            ap = 1 / rank
            self.mAP_total += ap
        
        return loss

    def on_test_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes,axis=0)
        all_true_labels = np.array(self.all_true_labels)
        
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct)/len(top_1_correct)

        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct)/len(top_k_correct)

        self.mAP = (self.mAP_total / len(all_true_labels)).item()
        self.match_similarities = np.mean(self.match_similarities) if self.match_similarities else 0

        
        self.log('test_top1_acc', top_1_accuracy, sync_dist=True)
        self.log('test_top5_acc', top_k_accuracy, sync_dist=True)
        self.log('mAP', self.mAP, sync_dist=True)
        self.log('similarity', self.match_similarities, sync_dist=True)
        self.log('score', self.score, sync_dist=True)


        self.all_predicted_classes = []
        self.all_true_labels = []

        avg_test_loss = self.trainer.callback_metrics['test_loss']
        return  {'test_loss': avg_test_loss.item(), 'test_top1_acc': top_1_accuracy.item(),'test_top5_acc':top_k_accuracy.item(),'mAP':self.mAP,'similarity':self.match_similarities, 'score': self.score}


    def configure_optimizers(self):

        optimizer_class = globals()[self.config['train']['optimizer']]
        optimizer = optimizer_class(self.parameters(), lr=self.config['train']['lr'], weight_decay=1e-4)
        return [optimizer]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/SIMON_temp.yaml", help="path to config")
    parser.add_argument("--seed", type=int, default=42, help="the seed")
    parser.add_argument("--exp_setting", type=str, default='intra-subject', help="exp_setting") #inter-subject 
    parser.add_argument("--epoch", type=int, default=50, help="train epoch")
    parser.add_argument("--lr", type=float, default=3e-4, help="lr")   # default = 3e-5, for inter setting
    parser.add_argument("--brain_backbone", type=str, default="EEGProjectLayer", help="brain_backbone")  #TSconv, for inter setting
    parser.add_argument("--vision_backbone", type=str, default="RN50", help="vision_backbone")
    parser.add_argument("--c", type=int, default=6, help="c")
    parser.add_argument("--selected_ch", type=str, default="OP", help="EEG selected channels") #ALL, for inter setting
    parser.add_argument("--eeg_data_module", type=str, default="data_temp", help="EEG data module")
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument("--residual_strength", type=float, default=1.0)

    opt = parser.parse_args()
    
    # Setup GPU environment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    config = update_config(opt, config)

    if opt.selected_ch is not None:
        config['data']['selected_ch'] = opt.selected_ch

    # Check CUDA availability
    if torch.cuda.is_available():
        accelerator = "cuda"
    else:
        accelerator = "cpu"
        print(f"--- Warning: GPU not available. Falling back to CPU ---")

    torch.set_float32_matmul_precision('high')
    
    top1_acc_list = []
    top5_acc_list = []
    mac_list = []
    score_list = []
    
    # Set channel number based on selection
    ch_map = {
        'ALL': 63, 'O': 8, 'P': 9, 'C': 14, 'T': 10, 'F': 22, 'OP': 17, 'OPT': 27, 'OPCT': 41
    }
    config['c_num'] = ch_map.get(config['data']['selected_ch'], 17) # default to 17 if not found
    config['models']['brain']['params']['c_num'] = int(config['c_num'])

    if str(opt.brain_backbone).endswith("_g"):
        config['models']['brain']['params']['residual_strength'] = float(opt.residual_strength)

    data_module_name = f"base.{opt.eeg_data_module}"
    print(f"[INFO] Using EEG data module: {data_module_name}")
    data_module = importlib.import_module(data_module_name)
    load_data = data_module.load_data
    sub_list = range(1, 11)  

    base_name = config['name']
    
    pretrain_map = {
        'RN50': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 1024},
        'RN101': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-16': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-32': {'pretrained': 'laion2b_s34b_b79k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-L-14': {'pretrained': 'laion2b_s32b_b82k', 'resize': (224, 224), 'z_dim': 768},
        'ViT-H-14': {'pretrained': 'laion2b_s32b_b79k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-g-14': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-bigG-14': {'pretrained': 'laion2b_s39b_b160k', 'resize': (224, 224), 'z_dim': 1280},
    }

    for subject_idx in sub_list:
        subject_str = f"sub-{str(subject_idx).zfill(2)}"
        print('subject_idx: ', subject_str)
        config['data']['subjects'] = [subject_str]

        # Config setup
        config['z_dim'] = pretrain_map[opt.vision_backbone]['z_dim']
        config['vision_backbone'] = opt.vision_backbone
        print(config)

        os.makedirs(config['save_dir'], exist_ok=True)
        version_tag = f"{'_'.join(config['data']['subjects'])}_seed{config['seed']}"          

        logger = TensorBoardLogger(config['save_dir'], name=config['name'], version=version_tag)
        os.makedirs(logger.log_dir, exist_ok=True)
        shutil.copy(opt.config, os.path.join(logger.log_dir, opt.config.rsplit('/', 1)[-1]))

        train_loader, val_loader, test_loader = load_data(config)
        
        pl_model = load_model(config, train_loader, test_loader)

        checkpoint_callback = ModelCheckpoint(save_last=True)

        if config['exp_setting'] == 'inter-subject':
            early_stop_callback = EarlyStopping(
                monitor='val_top1_acc',
                min_delta=0.001,     
                patience=5, 
                verbose=False,
                mode='max' 
            )
        else:
            early_stop_callback = EarlyStopping(
                monitor='train_loss',
                min_delta=0.001,
                patience=5,
                verbose=False,  
                mode='min' 
            )

        trainer = Trainer(
            log_every_n_steps=10,
            callbacks=[early_stop_callback, checkpoint_callback],
            max_epochs=config['train']['epoch'],
            devices=1,
            accelerator=accelerator,
            logger=logger
        )
        print(trainer.logger.log_dir)

        ckpt_path = None 

        trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)

        if config['exp_setting'] == 'inter-subject':
            test_results = trainer.test(ckpt_path='best', dataloaders=test_loader)
        else:
            test_results = trainer.test(ckpt_path='last', dataloaders=test_loader)

        with open(os.path.join(logger.log_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=4)

        top1_acc_list.append(test_results[0]['test_top1_acc'])  
        top5_acc_list.append(test_results[0]['test_top5_acc'])
        mac_list.append(test_results[0]['mAP'])
        score_list.append(test_results[0]['score'])

        
    print('Mean Top-1 Acc:', np.array(top1_acc_list).mean())
    print(top1_acc_list)
    print('Mean Top-5 Acc:', np.array(top5_acc_list).mean())
    print(top5_acc_list)
    print('Mean mAP:', np.array(mac_list).mean())
    print(mac_list)
    print('Mean Score:', np.array(score_list).mean())
    print(score_list)

if __name__ == "__main__":
    main()

