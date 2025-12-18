#!/bin/bash

LEARNING_RATE_VALUES=(0.001 0.0005 0.0002 0.0001 0.00005 0.00002 0.00001)

for LEARNING_RATE in "${LEARNING_RATE_VALUES[@]}"; do
    echo "Running with LEARNING_RATE=${LEARNING_RATE}"
    LEARNING_RATE=${LEARNING_RATE} bash /home/haris/project/predictor/scripts/run_neural_network_parallel_process_data.sh
    echo "Finished LEARNING_RATE=${LEARNING_RATE}"
    echo ""
done