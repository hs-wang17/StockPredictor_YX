#!/bin/bash

# ==================================================
# Run Neural Network Predictor Pipeline
# ==================================================

# -----------------------------
# Default parameters (can be overridden by ENV variables)
# -----------------------------
: "${DATA_DIR:=/home/haris/mydata/concat_daily_factor}"
: "${DEVICE:=cuda:0}"
: "${EPOCHS:=200}"
: "${FILTER_FILE_PATH:=config/filter_index.fea}"
: "${GAP_DAYS:=20}"
: "${HIDDEN_DIM:=64}"
: "${K_FOLDS:=4}"
: "${LABEL_FILE_PATH:=/home/haris/mydata/label.fea}"
: "${LEARNING_RATE:=1e-4}"
: "${LR_DECAY_GAMMA:=0.99}"
: "${LOG_DIR:=/home/haris/results/logs}"
: "${MODEL_SAVE_DIR:=/home/haris/results/models}"
: "${NUM_PERIODS:=}"
: "${PREDICT_BATCH_SIZE:=64}"
: "${PREDICT_PERIOD_DAYS:=60}"
: "${PREDICTIONS_SAVE_DIR:=/home/haris/results/predictions}"
: "${PROJECT_NAME:=StockPredictor}"
: "${MODEL_SAVE_FREQUENCY:=20}"
: "${SLIDE_PERIOD_DAYS:=60}"
: "${TRAIN_BATCH_SIZE:=1}"
: "${TRAIN_PERIOD_DAYS:=720}"
: "${USE_SWANLAB:=True}"

# -----------------------------
# Run Python script
# -----------------------------
python /home/haris/project/predictor/src/main_neural_network.py \
    --data_dir "${DATA_DIR}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --filter_file_path "${FILTER_FILE_PATH}" \
    --gap_days "${GAP_DAYS}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --k_folds "${K_FOLDS}" \
    --label_file_path "${LABEL_FILE_PATH}" \
    --learning_rate "${LEARNING_RATE}" \
    --lr_decay_gamma "${LR_DECAY_GAMMA}" \
    --log_dir "${LOG_DIR}" \
    --model_save_dir "${MODEL_SAVE_DIR}" \
    --predict_batch_size "${PREDICT_BATCH_SIZE}" \
    --predict_period_days "${PREDICT_PERIOD_DAYS}" \
    --predictions_save_dir "${PREDICTIONS_SAVE_DIR}" \
    --project_name "${PROJECT_NAME}" \
    --model_save_frequency "${MODEL_SAVE_FREQUENCY}" \
    --slide_period_days "${SLIDE_PERIOD_DAYS}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --train_period_days "${TRAIN_PERIOD_DAYS}" \
    --use_swanlab "${USE_SWANLAB}" \
    ${NUM_PERIODS:+--num_periods "${NUM_PERIODS}"}

# -----------------------------
# Finish message
# -----------------------------
echo "Neural Network predictor pipeline finished!"
