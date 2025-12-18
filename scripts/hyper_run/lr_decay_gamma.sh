#!/bin/bash

LR_DECAY_GAMMA_VALUES=(0.999 0.995 0.99 0.95 0.9 0.85 0.8)

for LR_DECAY_GAMMA in "${LR_DECAY_GAMMA_VALUES[@]}"; do
    echo "Running with LR_DECAY_GAMMA=${LR_DECAY_GAMMA}"
    LR_DECAY_GAMMA=${LR_DECAY_GAMMA} bash /home/haris/project/predictor/scripts/run_neural_network_parallel_process_data.sh
    echo "Finished LR_DECAY_GAMMA=${LR_DECAY_GAMMA}"
    echo ""
done