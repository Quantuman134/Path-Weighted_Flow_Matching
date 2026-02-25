#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2  # Set this to the GPU(s) you want to use (e.g., "0,1" for multiple GPUs)
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# SiT Training Script
# Configuration file path
config_file="./configs/sit_config.yaml"

# Number of GPUs to use
NUM_GPUS=2

# Run training with torchrun (PyTorch distributed)
torchrun --nproc_per_node=$NUM_GPUS train.py --config $config_file
