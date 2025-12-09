#!/bin/bash

# ==================================================
# Run LightGBM Predictor Pipeline
# ==================================================

# -----------------------------
# Default parameters (can be overridden by ENV variables)
# -----------------------------
: "${BEGIN_PERIOD:=0}"
: "${DATA_DIR:=/home/user0/mydata/concat_daily_factor_with_label}"
: "${DEVICE:=cuda}"
: "${END_DATE:=20251024}"
: "${FILTER_FILE_PATH:=config/filter_index.fea}"
: "${TRAIN_PERIOD_DAYS:=720}"
: "${PREDICT_PERIOD_DAYS:=60}"
: "${GAP_DAYS:=20}"
: "${K_FOLDS:=4}"
: "${LOG_DIR:=/home/user0/results/logs}"
: "${MODEL_SAVE_DIR:=/home/user0/results/models}"
: "${PREDICTIONS_SAVE_DIR:=/home/user0/results/predictions}"
: "${PROJECT_NAME:=StockPredictor}"
: "${SLIDE_PERIOD_DAYS:=60}"
: "${NUM_PERIODS:=}"
: "${START_DATE:=20210101}"
: "${USE_SWANLAB:=True}"

# -----------------------------
# LightGBM-specific parameters
# -----------------------------
: "${N_ESTIMATORS:=10000}"
: "${OBJECTIVE:=regression}"
: "${BOOSTING_TYPE:=gbdt}"
: "${RANDOM_STATE:=42}"
: "${LEARNING_RATE:=0.05}"
: "${NUM_LEAVES:=255}"
: "${MIN_DATA_IN_LEAF:=20}"
: "${FEATURE_FRACTION:=0.1}"
: "${BAGGING_FRACTION:=0.8}"
: "${BAGGING_FREQ:=1}"
: "${EARLY_STOPPING_ROUNDS:=500}"
: "${VALID_SIZE:=0.1}"
: "${VERBOSE_EVAL:=100}"

# -----------------------------
# Run Python script
# -----------------------------
python /home/user0/project/predictor/src/main_ensemble.py \
    --begin_period "${BEGIN_PERIOD}" \
    --data_dir "${DATA_DIR}" \
    --device "${DEVICE}" \
    --end_date "${END_DATE}" \
    --filter_file_path "${FILTER_FILE_PATH}" \
    --train_period_days "${TRAIN_PERIOD_DAYS}" \
    --predict_period_days "${PREDICT_PERIOD_DAYS}" \
    --gap_days "${GAP_DAYS}" \
    --k_folds "${K_FOLDS}" \
    --log_dir "${LOG_DIR}" \
    --model_save_dir "${MODEL_SAVE_DIR}" \
    --predictions_save_dir "${PREDICTIONS_SAVE_DIR}" \
    --project_name "${PROJECT_NAME}" \
    --slide_period_days "${SLIDE_PERIOD_DAYS}" \
    --start_date "${START_DATE}" \
    --use_swanlab "${USE_SWANLAB}" \
    --n_estimators "${N_ESTIMATORS}" \
    --objective "${OBJECTIVE}" \
    --boosting_type "${BOOSTING_TYPE}" \
    --random_state "${RANDOM_STATE}" \
    --learning_rate "${LEARNING_RATE}" \
    --num_leaves "${NUM_LEAVES}" \
    --min_data_in_leaf "${MIN_DATA_IN_LEAF}" \
    --feature_fraction "${FEATURE_FRACTION}" \
    --bagging_fraction "${BAGGING_FRACTION}" \
    --bagging_freq "${BAGGING_FREQ}" \
    --early_stopping_rounds "${EARLY_STOPPING_ROUNDS}" \
    --valid_size "${VALID_SIZE}" \
    --verbose_eval "${VERBOSE_EVAL}" \
    ${NUM_PERIODS:+--num_periods "${NUM_PERIODS}"}

# -----------------------------
# Finish message
# -----------------------------
echo "LightGBM predictor pipeline finished!"
