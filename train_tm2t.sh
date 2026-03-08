#!/bin/bash

CUDA_VISIBLE_DEVICES=3  # Set this to the GPU(s) you want to use
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=16

# TM2T Training Script
# Configuration file path
config_file="./configs/tm2t_config.yaml"

# Number of GPUs to use
NUM_GPUS=1

# Run training with torchrun (PyTorch distributed)
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
NCCL_P2P_DISABLE=1 torchrun --master_port=$MASTER_PORT --nproc_per_node=$NUM_GPUS train_tm2t.py --config $config_file
