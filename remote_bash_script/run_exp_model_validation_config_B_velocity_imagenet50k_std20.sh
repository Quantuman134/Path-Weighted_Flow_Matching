#!/bin/bash
# Validation sweep: SiT-B/2, baseline (unweighted velocity MSE), 50k ImageNet subset with latent sigma 2.0
source /scratch/project/prj-02-visual-ai/hkzhang/miniconda3/etc/profile.d/conda.sh
conda activate SiT
cd /scratch/project/prj-02-visual-ai/hkzhang/Path-Weighted_Flow_Matching
NUM_GPUS=8
SINGLE_GPU_DEVICE="cuda:0"
CONFIG="configs/exp_model_validation_config_B_velocity_imagenet50k_std20.yaml"
MASTER_PORT=$(( RANDOM % 3268 + 29500 ))

echo "════════════════════════════════════════════════════════"
echo " Config : $CONFIG"
echo " GPUs   : $NUM_GPUS"
echo " Port   : $MASTER_PORT"
echo "════════════════════════════════════════════════════════"

if [ "$NUM_GPUS" -gt 1 ]; then
    NCCL_P2P_DISABLE=1 torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port="${MASTER_PORT}" \
        exp_model_validation.py \
        --config "${CONFIG}"
else
    python exp_model_validation.py \
        --config "${CONFIG}" \
        --device "${SINGLE_GPU_DEVICE}"
fi
