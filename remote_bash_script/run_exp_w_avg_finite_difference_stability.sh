#!/bin/bash
# Experiment A: finite-difference stability check for w_avg(t) on pretrained SiT-XL/2.
# Sweeps eta in {1e-5..1e-2} at t in {0, 0.5, 0.9} to pick the largest stable eta.
source /scratch/project/prj-02-visual-ai/hkzhang/miniconda3/etc/profile.d/conda.sh
conda activate SiT
cd /scratch/project/prj-02-visual-ai/hkzhang/Path-Weighted_Flow_Matching

NUM_GPUS=8
SINGLE_GPU_DEVICE="cuda:0"
CONFIG="configs/exp_w_avg_finite_difference_stability.yaml"
MASTER_PORT=$(( RANDOM % 3268 + 29500 ))

echo "════════════════════════════════════════════════════════"
echo " Experiment  : w_avg finite-difference STABILITY (Experiment A)"
echo " Config      : $CONFIG"
echo " GPUs        : $NUM_GPUS"
echo " Port        : $MASTER_PORT"
echo "════════════════════════════════════════════════════════"

if [ "$NUM_GPUS" -gt 1 ]; then
    NCCL_P2P_DISABLE=1 torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port="${MASTER_PORT}" \
        exp_w_avg_finite_difference.py \
        --config "${CONFIG}"
else
    python exp_w_avg_finite_difference.py \
        --config "${CONFIG}" \
        --device "${SINGLE_GPU_DEVICE}"
fi
