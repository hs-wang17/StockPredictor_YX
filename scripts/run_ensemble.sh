#!/bin/bash

# ==================================================
# Run LightGBM Predictor Pipeline
# ==================================================

# Default parameters (can be overridden by passing arguments)
DATA_DIR="/home/user0/data/StockDailyData/"
DEVICE="cpu"   # LightGBM 默认使用 CPU，若 GPU 可用并支持则改为 "gpu"
TRAIN_PERIOD_DAYS=720
PREDICT_PERIOD_DAYS=60
GAP_DAYS=20
LOG_DIR="/home/user0/results/logs"
MODEL_SAVE_DIR="/home/user0/results/models"
PREDICTIONS_SAVE_DIR="/home/user0/results/predictions"
PROJECT_NAME="StockPredictor"
SLIDE_PERIOD_DAYS=60
FILTER_FILE_PATH="config/filter_index.fea"
NUM_PERIODS=""

# LightGBM-specific parameters
LGB_N_ESTIMATORS=1000
EARLY_STOPPING_ROUNDS=50
VALID_SIZE=0.1
VERBOSE_EVAL=50
RANDOM_SEED=42
LGB_PARAMS="{}"  # JSON string of additional LightGBM params, e.g. '{"num_leaves":63,"learning_rate":0.01}'

# You can override any variable by passing ENV vars, e.g.:
# DATA_DIR=/path/to/data ./run_predictor_lgbm.sh

python /home/user0/project/predictor/src/main_lgbm.py \
    --data_dir "${DATA_DIR}" \
    --device "${DEVICE}" \
    --train_period_days "${TRAIN_PERIOD_DAYS}" \
    --predict_period_days "${PREDICT_PERIOD_DAYS}" \
    --gap_days "${GAP_DAYS}" \
    --log_dir "${LOG_DIR}" \
    --model_save_dir "${MODEL_SAVE_DIR}" \
    --predictions_save_dir "${PREDICTIONS_SAVE_DIR}" \
    --project_name "${PROJECT_NAME}" \
    --slide_period_days "${SLIDE_PERIOD_DAYS}" \
    --filter_file_path "${FILTER_FILE_PATH}" \
    --lgb_n_estimators "${LGB_N_ESTIMATORS}" \
    --early_stopping_rounds "${EARLY_STOPPING_ROUNDS}" \
    --valid_size "${VALID_SIZE}" \
    --verbose_eval "${VERBOSE_EVAL}" \
    --random_seed "${RANDOM_SEED}" \
    --lgb_params "${LGB_PARAMS}" \
    ${NUM_PERIODS:+--num_periods "${NUM_PERIODS}"}

echo "LightGBM predictor pipeline finished!"
