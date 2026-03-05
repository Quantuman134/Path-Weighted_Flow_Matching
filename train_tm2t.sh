#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2  # Set this to the GPU(s) you want to use
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=16

# TM2T Training Script
# Configuration file path
config_file="./configs/tm2t_config.yaml"

# Number of GPUs to use
NUM_GPUS=2

# Run training with torchrun (PyTorch distributed)
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=$NUM_GPUS --master_port=29507 train_tm2t.py --config $config_file
