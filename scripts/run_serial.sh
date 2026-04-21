#!/bin/bash

# 定义基础路径
SCRIPT_BASE="/home/haris/project/predictor/scripts/save"
LOG_BASE="/home/haris/logs"

# 定义需要运行的脚本列表（按日期顺序）
scripts=(
    "run_neural_network_parallel_process_data_training_batch_20260421_noon.sh"
    "run_neural_network_parallel_process_data_training_batch_20260422_noon.sh"
    "run_neural_network_parallel_process_data_training_batch_20260423_noon.sh"
    "run_neural_network_parallel_process_data_training_batch_20260424_noon.sh"
    "run_neural_network_parallel_process_data_training_batch_20260425_noon.sh"
    "run_neural_network_parallel_process_data_training_batch_20260426_noon.sh"
    "run_neural_network_parallel_process_data_training_batch_20260427_noon.sh"
    "run_neural_network_parallel_process_data_training_batch_20260428_noon.sh"
    "run_neural_network_parallel_process_data_training_batch_20260429_noon.sh"
    "run_neural_network_parallel_process_data_training_batch_20260430_noon.sh"
    "run_neural_network_parallel_process_data_training_batch_20260501_noon.sh"
    "run_neural_network_parallel_process_data_training_batch_20260502_noon.sh"
    "run_neural_network_parallel_process_data_training_batch_20260503_noon.sh"
    "run_neural_network_parallel_process_data_training_batch_20260504_noon.sh"
    "run_neural_network_parallel_process_data_training_batch_20260505_noon.sh"
    "run_neural_network_parallel_process_data_training_batch_20260506_noon.sh"
    "run_neural_network_parallel_process_data_training_batch_20260507_noon.sh"
)

# 循环串行执行每个脚本
for script in "${scripts[@]}"; do
    # 提取日期部分用于日志命名（例如 20260420）
    date_part=$(echo "$script" | grep -oP '\d{8}')
    
    # 构建完整脚本路径和日志路径
    full_script="${SCRIPT_BASE}/${script}"
    log_file="${LOG_BASE}/run_para_${date_part}.log"
    
    echo "Starting: $script"
    
    # 创建 screen 会话并运行脚本（等待完成后再继续下一个）
    screen -dmS "stock_${date_part}" bash -c "bash '$full_script' > '$log_file' 2>&1"
    
    # 等待当前 screen 会话结束（串行控制）
    while screen -list | grep -q "stock_${date_part}"; do
        sleep 10
    done
    
    echo "Finished: $script (log: $log_file)"
done

echo "All scripts executed successfully."