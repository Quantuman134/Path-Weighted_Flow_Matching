#!/bin/bash

# SiT Training Script
# Configuration file path
config_file="./configs/sit_config.yaml"

# Number of GPUs to use
NUM_GPUS=8

# Run training with torchrun (PyTorch distributed)
torchrun --nproc_per_node=$NUM_GPUS train.py --config $config_file
