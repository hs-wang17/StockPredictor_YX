#!/bin/bash

# ==================================================
# Run Neural Network Predictor Pipeline
# ==================================================

# -----------------------------
# Default parameters (can be overridden by ENV variables)
# -----------------------------
: "${BEGIN_PERIOD:=0}"
: "${DATA_DIR:=/home/haris/raid0/shared/haris/mydata_20251231/concat_daily_factor}"
: "${DEVICE:=cpu}"
: "${END_DATE:=}"
: "${EPOCHS:=80}"
: "${FILTER_FILE_PATH:=config/filter_index.fea}"
: "${FROM_START:=False}"
: "${GAP_DAYS:=10}"
: "${HIDDEN_DIM:=64}"
: "${INVERSE:=False}"
: "${K_FOLDS:=4}"
# : "${LABEL_FILE_PATH:=/home/haris/mydata/label.fea}"
: "${LEARNING_RATE:=0.0001}"
: "${LR_DECAY_GAMMA:=0.99}"
: "${LOG_DIR:=/home/haris/results/logs}"
: "${MODEL_TYPE:=mlp}"
: "${MODEL_SAVE_DIR:=/home/haris/mymodel_10/models/StockPredictor_20260306}"
: "${NUM_PERIODS:=}"
: "${PREDICT_BATCH_SIZE:=0}"
: "${PREDICT_PERIOD_DAYS:=0}"
: "${PREDICTIONS_SAVE_DIR:=/home/haris/mymodel_10/predictions/StockPredictor_20260306}"
: "${PROJECT_NAME:=StockPredictor_20260306}"
: "${MODEL_SAVE_FREQUENCY:=1}"
: "${SLIDE_PERIOD_DAYS:=0}"
: "${START_DATE:=20260101}"
: "${SUFFIX:=_20260306}"
: "${TRADE_DATE_PATH:=/home/haris/raid0/shared/haris/mydata_20251231/trade_date_10.fea}"
: "${TRAIN_BATCH_SIZE:=1}"
: "${TRAIN_PERIOD_DAYS:=720}"
: "${USE_SWANLAB:=False}"

# -----------------------------
# Run Python script
# -----------------------------
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/project/predictor/src/update_neural_network_parallel_predict.py \
    --begin_period "${BEGIN_PERIOD}" \
    --data_dir "${DATA_DIR}" \
    --device "${DEVICE}" \
    --end_date "${END_DATE}" \
    --epochs "${EPOCHS}" \
    --filter_file_path "${FILTER_FILE_PATH}" \
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
    --model_save_frequency "${MODEL_SAVE_FREQUENCY}" \
    --slide_period_days "${SLIDE_PERIOD_DAYS}" \
    --start_date "${START_DATE}" \
    --suffix "${SUFFIX}" \
    --trade_date_path "${TRADE_DATE_PATH}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --train_period_days "${TRAIN_PERIOD_DAYS}" \
    --use_swanlab "${USE_SWANLAB}" \
    ${NUM_PERIODS:+--num_periods "${NUM_PERIODS}"}

# -----------------------------
# Merge prediction data
# -----------------------------
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mymodel_10/merge_data_20260306_all_stocks.py

# -----------------------------
# Finish message
# -----------------------------
echo "Neural Network predictor pipeline finished!"
# cat /home/haris/logs/update_predict_20260306.log | mail -s "每日预测任务日志(morning_10) - $(date +\%Y-\%m-\%d)" xsheng9867@163.com