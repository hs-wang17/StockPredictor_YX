#!/bin/bash

# ==================================================
# Run Neural Network Predictor Pipeline
# ==================================================

# -----------------------------
# Default parameters (can be overridden by ENV variables)
# -----------------------------
: "${BEGIN_PERIOD:=0}"
: "${CRITERION:=mse}"
: "${DATA_DIR:=/home/haris/raid0/shared/haris/mydata_20260127/concat_daily_factor_with_label_9}"
: "${DEVICE:=cuda:0}"
: "${END_DATE:=20301231}"
: "${EPOCHS:=40}"
: "${FILTER_FILE_PATH:=/home/haris/raid0/shared/haris/mydata_20260127/feature_selection/selected_factor_index_rankic_correlation_matrix_10.csv}"
: "${FROM_START:=False}"
: "${GAP_DAYS:=9}"
: "${HIDDEN_DIM:=64}"
: "${INVERSE:=False}"
: "${K_FOLDS:=4}"
# : "${LABEL_FILE_PATH:=/home/haris/mydata/label.fea}"
: "${LEARNING_RATE:=0.0001}"
: "${LR_DECAY_GAMMA:=0.99}"
: "${LOG_DIR:=/home/haris/results/logs}"
: "${MODEL_TYPE:=resnet}"
: "${MODEL_SAVE_DIR:=/home/haris/results/models}"
: "${NUM_PERIODS:=}"
: "${PREDICT_BATCH_SIZE:=64}"
: "${PREDICT_PERIOD_DAYS:=60}"
: "${PREDICTIONS_SAVE_DIR:=/home/haris/results/predictions}"
: "${PROJECT_NAME:=StockPredictor}"
: "${REMOVE_ABNORMAL:=False}"
: "${MODEL_SAVE_FREQUENCY:=20}"
: "${SLIDE_PERIOD_DAYS:=60}"
: "${START_DATE:=20190101}"
: "${TRADE_DATE_PATH:=/home/haris/raid0/shared/haris/mydata_20260127/trade_date_9.fea}"
: "${TRAIN_BATCH_SIZE:=1}"
: "${TRAIN_PERIOD_DAYS:=720}"
: "${USE_SWANLAB:=True}"


# -----------------------------
# Run Python script
# -----------------------------
python /home/haris/project/predictor/src/main_neural_network_parallel_process_data_training_batch.py \
    --begin_period "${BEGIN_PERIOD}" \
    --criterion "${CRITERION}" \
    --data_dir "${DATA_DIR}" \
    --device "${DEVICE}" \
    --end_date "${END_DATE}" \
    --epochs "${EPOCHS}" \
    --from_start "${FROM_START}" \
    --gap_days "${GAP_DAYS}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --inverse "${INVERSE}" \
    --k_folds "${K_FOLDS}" \
    --learning_rate "${LEARNING_RATE}" \
    --lr_decay_gamma "${LR_DECAY_GAMMA}" \
    --log_dir "${LOG_DIR}" \
    --model_type "${MODEL_TYPE}" \
    --model_save_dir "${MODEL_SAVE_DIR}" \
    --predict_batch_size "${PREDICT_BATCH_SIZE}" \
    --predict_period_days "${PREDICT_PERIOD_DAYS}" \
    --predictions_save_dir "${PREDICTIONS_SAVE_DIR}" \
    --project_name "${PROJECT_NAME}" \
    --remove_abnormal "${REMOVE_ABNORMAL}" \
    --model_save_frequency "${MODEL_SAVE_FREQUENCY}" \
    --slide_period_days "${SLIDE_PERIOD_DAYS}" \
    --start_date "${START_DATE}" \
    --trade_date_path "${TRADE_DATE_PATH}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --train_period_days "${TRAIN_PERIOD_DAYS}" \
    --use_swanlab "${USE_SWANLAB}" \
    ${NUM_PERIODS:+--num_periods "${NUM_PERIODS}"}

# -----------------------------
# Finish message
# -----------------------------
echo "Neural Network predictor pipeline finished!"
