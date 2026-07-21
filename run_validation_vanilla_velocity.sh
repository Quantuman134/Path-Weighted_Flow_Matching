#!/bin/bash
# ============================================================
# run_validation_vanilla_velocity.sh
#
# Runs FID/IS evaluation for:
#   ImageNet:  SiT-S/L/XL  vanilla_weighting_v lam=1.0
#   CIFAR-10:  SiT-S/B/L/XL vanilla_weighting_v lam=0.5
#   CIFAR-10:  SiT-S/B/L/XL velocity
#
# Edit NUM_GPUS, then:
#   bash run_validation_vanilla_velocity.sh
# ============================================================

CONFIGS=(
    # ── ImageNet ─────────────────────────────────────────────
    "configs/exp_model_validation_config_S_vanilla_10_imagenet.yaml"
    "configs/exp_model_validation_config_L_vanilla_10_imagenet.yaml"
    "configs/exp_model_validation_config_XL_vanilla_10_imagenet_finetuning.yaml"

    # ── CIFAR-10  vanilla_weighting_v lam=0.5 ────────────────
    "configs/exp_model_validation_config_S_vanilla_05_cifar10.yaml"
    "configs/exp_model_validation_config_B_vanilla_05_cifar10.yaml"
    "configs/exp_model_validation_config_L_vanilla_05_cifar10.yaml"
    "configs/exp_model_validation_config_XL_vanilla_05_cifar10.yaml"

    # ── CIFAR-10  velocity ────────────────────────────────────
    "configs/exp_model_validation_config_S_velocity_cifar10.yaml"
    "configs/exp_model_validation_config_B_velocity_cifar10.yaml"
    "configs/exp_model_validation_config_L_velocity_cifar10.yaml"
    "configs/exp_model_validation_config_XL_velocity_cifar10.yaml"
)

NUM_GPUS=8

SINGLE_GPU_DEVICE="cuda:0"

# ── Run each config sequentially ─────────────────────────────────────────────
for CONFIG in "${CONFIGS[@]}"; do
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
