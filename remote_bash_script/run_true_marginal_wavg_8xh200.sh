#!/bin/bash
# Analytic marginal w_avg(T, t) on the empirical ImageNet latent distribution
# (revision v2, 2026-08-03).
#
# See tmp/true_marginal_wavg_required_revisions.html and
# exp_true_marginal_wavg.py.
#
# The main run shards C=32 classes round-robin across 8 H200 GPUs
# (4 classes per rank). Only the final aggregation and the validation
# subrun happen on rank 0.
#
# To run a sweep at (T, S) other than the config default, edit the config
# fields flow.terminal_time / flow.solver_steps and either pass a new
# --output dir on the second CLI arg, or set output_dir inside the config.

set -euo pipefail

source /scratch/project/prj-02-visual-ai/hkzhang/miniconda3/etc/profile.d/conda.sh
conda activate SiT
cd /scratch/project/prj-02-visual-ai/hkzhang/Path-Weighted_Flow_Matching

NUM_GPUS=8
SINGLE_GPU_DEVICE="cuda:0"
CONFIG="${1:-configs/true_marginal_wavg_imagenet.yaml}"
OUT="${2:-experiment/true_marginal_wavg_imagenet_T099_S256}"
MASTER_PORT=$(( RANDOM % 3268 + 29500 ))

mkdir -p "${OUT}"
nvidia-smi | tee "${OUT}/nvidia_smi.txt" > /dev/null

echo "════════════════════════════════════════════════════════"
echo " Experiment  : true_marginal_wavg_imagenet (revision v2, T<1 fixed)"
echo " Config      : $CONFIG"
echo " Output dir  : $OUT"
echo " GPUs        : $NUM_GPUS"
echo " Port        : $MASTER_PORT"
echo "════════════════════════════════════════════════════════"

if [ "$NUM_GPUS" -gt 1 ]; then
    NCCL_P2P_DISABLE=1 torchrun \
        --standalone \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port="${MASTER_PORT}" \
        exp_true_marginal_wavg.py \
        --config "${CONFIG}" \
        --output "${OUT}" \
        2>&1 | tee -a "${OUT}/run.log"
else
    python exp_true_marginal_wavg.py \
        --config "${CONFIG}" \
        --output "${OUT}" \
        --device "${SINGLE_GPU_DEVICE}" \
        2>&1 | tee -a "${OUT}/run.log"
fi
