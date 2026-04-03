#!/bin/bash
# ============================================================
# run_validation.sh — Launch exp_model_validation.py
#
# Edit CONFIG and NUM_GPUS before running, then:
#   bash run_validation.sh
# ============================================================

# Path to the config file
CONFIG="configs/exp_model_validation_config.yaml"

# Number of GPUs to use.
# Set to 1 for single-GPU mode (uses --device from config or CLI).
# Set to >1 for multi-GPU mode (uses torchrun with NCCL).
NUM_GPUS=8

# (Optional) Override the device for single-GPU mode.
# Ignored when NUM_GPUS > 1.
SINGLE_GPU_DEVICE="cuda:0"

# Randomly pick a master port in [29500, 32767] to avoid conflicts
MASTER_PORT=$(( RANDOM % 3268 + 29500 ))

# ── Launch ────────────────────────────────────────────────────────────────────
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching with torchrun on ${NUM_GPUS} GPUs..."
    NCCL_P2P_DISABLE=1 torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port="${MASTER_PORT}" \
        exp_model_validation.py \
        --config "${CONFIG}"
else
    echo "Launching on single GPU: ${SINGLE_GPU_DEVICE}..."
    python exp_model_validation.py \
        --config "${CONFIG}" \
        --device "${SINGLE_GPU_DEVICE}"
fi
