#!/bin/bash
# ============================================================
# run_validation.sh — Launch exp_model_validation.py
#
# Add config file paths to CONFIGS to run multiple experiments
# sequentially. Each config runs to completion before the next.
#
# Edit CONFIGS and NUM_GPUS, then:
#   bash run_validation.sh
# ============================================================

# List of config files to run sequentially.
# Add or comment out entries as needed.
CONFIGS=(
    # "configs/exp_model_validation_config_linear_blend.yaml"
    # "configs/exp_model_validation_config_min_snr.yaml"
    # "configs/exp_model_validation_config_vv.yaml"
    "configs/exp_model_validation_config_tt.yaml"
    # "configs/exp_model_validation_config_2.yaml"
)

# Number of GPUs to use.
# Set to 1 for single-GPU mode.
# Set to >1 for multi-GPU mode (uses torchrun with NCCL).
NUM_GPUS=8

# Device for single-GPU mode (ignored when NUM_GPUS > 1).
SINGLE_GPU_DEVICE="cuda:0"

# ── Run each config sequentially ─────────────────────────────────────────────
for CONFIG in "${CONFIGS[@]}"; do
    # Fresh random port per run to avoid conflicts
    MASTER_PORT=$(( RANDOM % 3268 + 29500 ))

    echo ""
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

    echo ""
    echo "Finished: $CONFIG"
done

echo ""
echo "All configs complete."
