#!/bin/bash
source /scratch/project/prj-02-visual-ai/hkzhang/miniconda3/etc/profile.d/conda.sh
conda activate SiT
cd /scratch/project/prj-02-visual-ai/hkzhang/Path-Weighted_Flow_Matching

CUDA_VISIBLE_DEVICES="0, 1, 2, 3, 4, 5, 6, 7,"  # Set this to the GPU(s) you want to use (e.g., "0,1" for multiple GPUs)
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=16  # 256 CPU cores / 2 GPUs = 128, using 16 for good balance

# SiT Training Script
# Configuration file path
config_file="./configs/sit_config_S_vanilla_weighting_05_scale_cifar10.yaml"

# Number of GPUs to use
NUM_GPUS=8

# Run training with torchrun (PyTorch distributed)
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train.py --config $config_file