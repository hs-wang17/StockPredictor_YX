#!/bin/bash

# ==================================================
# Run Neural Network Predictor Pipeline
# ==================================================

# Default parameters (can be overridden by passing arguments)
DATA_DIR="/home/user0/data/StockDailyData/"
DEVICE="cuda"
TRAIN_PERIOD_DAYS=720
PREDICT_PERIOD_DAYS=60
GAP_DAYS=20
EPOCHS=10
TRAIN_BATCH_SIZE=64
PREDICT_BATCH_SIZE=1
LEARNING_RATE=0.001
LOG_DIR="/home/user0/results/logs"
MODEL_SAVE_DIR="/home/user0/results/models"
PREDICTIONS_SAVE_DIR="/home/user0/results/predictions"
PROJECT_NAME="StockPredictor"
MODEL_SAVE_FREQUENCY=5
SLIDE_PERIOD_DAYS=60
FILTER_FILE_PATH="config/filter_index.fea"
NUM_PERIODS=""

# You can override any variable by passing ENV vars, e.g.:
# DATA_DIR=/path/to/data ./run_predictor_nn.sh

python /home/user0/project/predictor/src/main_ensemble.py \
    --data_dir "${DATA_DIR}" \
    --device "${DEVICE}" \
    --train_period_days "${TRAIN_PERIOD_DAYS}" \
    --predict_period_days "${PREDICT_PERIOD_DAYS}" \
    --gap_days "${GAP_DAYS}" \
    --epochs "${EPOCHS}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --predict_batch_size "${PREDICT_BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --log_dir "${LOG_DIR}" \
    --model_save_dir "${MODEL_SAVE_DIR}" \
    --predictions_save_dir "${PREDICTIONS_SAVE_DIR}" \
    --project_name "${PROJECT_NAME}" \
    --model_save_frequency "${MODEL_SAVE_FREQUENCY}" \
    --slide_period_days "${SLIDE_PERIOD_DAYS}" \
    --filter_file_path "${FILTER_FILE_PATH}" \
    ${NUM_PERIODS:+--num_periods "${NUM_PERIODS}"}

echo "Neural Network predictor pipeline finished!"
