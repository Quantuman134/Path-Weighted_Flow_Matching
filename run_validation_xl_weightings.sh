#!/bin/bash
# ============================================================
# run_validation_xl_weightings.sh
#
# Runs FID/IS-vs-steps evaluation for the six SiT-XL/2 ImageNet-256
# timestep-weighting runs (training indices 040-045):
#
#   040  cosmap_v
#   041  kg_v
#   042  logit_normal_v
#   043  min_snr_gamma_v  (lam=5.0)
#   044  snr_v
#   045  rfpp_v
#
# Each config reads ./results/SiT-XL-2-...-imagenet/checkpoints/ (the
# training dirs with train.py's running index prefix stripped). All six
# share the step grid [50000, 100000, 150000, 200000] so the FID curves
# are directly comparable.
#
# Edit NUM_GPUS, then:
#   bash run_validation_xl_weightings.sh
# ============================================================

CONFIGS=(
    "configs/exp_model_validation_config_XL_cosmap_v_imagenet.yaml"
    "configs/exp_model_validation_config_XL_kg_v_imagenet.yaml"
    "configs/exp_model_validation_config_XL_logit_normal_v_imagenet.yaml"
    "configs/exp_model_validation_config_XL_min_snr_gamma_v_lam5_imagenet.yaml"
    "configs/exp_model_validation_config_XL_snr_v_imagenet.yaml"
    "configs/exp_model_validation_config_XL_rfpp_v_imagenet.yaml"
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
