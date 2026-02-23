#!/bin/bash

DATA_DIR="SIMON/data/things-eeg/"
EEG_PT_DIR="SIMON/data/things-eeg/Preprocessed_data_250Hz_whiten/sub-01"
OUT_DIR="Image_feature"
GPU_ID="0"

mkdir -p "$OUT_DIR"

echo "================================================================="
echo "Starting Feature Extraction Pipeline"
echo "Data Dir: $DATA_DIR"
echo "Output Dir: $OUT_DIR"
echo "================================================================="

echo "[1/4] Extracting High-Level Features (Train)..."
python Extract_embedding.py \
    --mode high \
    --eeg_pt "${EEG_PT_DIR}/train.pt" \
    --data_dir "$DATA_DIR" \
    --out_pt "${OUT_DIR}/high_level_train.pt" \
    --gpu "$GPU_ID"

echo "[2/4] Extracting High-Level Features (Test)..."
python Extract_embedding.py \
    --mode high \
    --eeg_pt "${EEG_PT_DIR}/test.pt" \
    --data_dir "$DATA_DIR" \
    --out_pt "${OUT_DIR}/high_level_test.pt" \
    --gpu "$GPU_ID"

echo "[3/4] Extracting Low-Level Features (Train)..."
python Extract_embedding.py \
    --mode low \
    --eeg_pt "${EEG_PT_DIR}/train.pt" \
    --data_dir "$DATA_DIR" \
    --out_pt "${OUT_DIR}/low_level_train.pt" \
    --blur_k 31 \
    --gpu "$GPU_ID"

echo "[4/4] Extracting Low-Level Features (Test)..."
python Extract_embedding.py \
    --mode low \
    --eeg_pt "${EEG_PT_DIR}/test.pt" \
    --data_dir "$DATA_DIR" \
    --out_pt "${OUT_DIR}/low_level_test.pt" \
    --blur_k 31 \
    --gpu "$GPU_ID"

echo "================================================================="
echo "All tasks completed successfully."
echo "================================================================="