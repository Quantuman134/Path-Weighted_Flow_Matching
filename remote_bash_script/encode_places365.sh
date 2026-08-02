#!/bin/bash
# Pre-encode Places365-Standard to VAE latents, then pack to per-class .npy.
#
#   1. encode  train/<class>/*.jpg -> latents/train/<class>/*.pt   (8 GPUs)
#   2. pack    latents/train       -> latents_packed/train/*.npy   (CPU)
#
# Expects the dataset already downloaded in ImageFolder layout:
#   wget -c http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
#   tar -xf places365standard_easyformat.tar
# If train/ contains single-letter dirs (a/, b/, ...) the categories are
# alphabet-nested and must be flattened to 365 top-level dirs first:
#   cd places365_standard/train && for d in ?/*/; do mv "$d" "$(echo ${d%/} | tr / _)"; done && rmdir ?
#
# 1.8M images: expect ~110 GB of .pt latents before packing. Both stages are
# resumable (existing .pt / .npy files are skipped), so re-run after a kill.
set -e

source /scratch/project/prj-02-visual-ai/hkzhang/miniconda3/etc/profile.d/conda.sh
conda activate SiT
cd /scratch/project/prj-02-visual-ai/hkzhang/Path-Weighted_Flow_Matching

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=16

DATA_ROOT=/scratch/project/prj-02-visual-ai/hkzhang/places365
TRAIN_DIR=$DATA_ROOT/places365_standard/train
LATENT_DIR=$DATA_ROOT/latents/train
PACKED_DIR=$DATA_ROOT/latents_packed/train

NUM_GPUS=8
IMAGE_SIZE=256
VAE=ema                  # matches model.vae in the config
BATCH_SIZE=64            # per-GPU
NUM_WORKERS=12           # per-GPU dataloader workers (1.8M small files on Lustre)
PACK_WORKERS=32

EXPECT_CLASSES=365       # Places365-Standard: 365 classes, 1,803,460 train images

# ── preconditions ────────────────────────────────────────────────────────────
if [ ! -d "$TRAIN_DIR" ]; then
    echo "ERROR: $TRAIN_DIR not found — download Places365-Standard first." >&2
    exit 1
fi

N_CLASSES=$(find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
if [ "$N_CLASSES" -ne "$EXPECT_CLASSES" ]; then
    echo "ERROR: found $N_CLASSES class dirs in $TRAIN_DIR, expected $EXPECT_CLASSES." >&2
    echo "       If this is 26, the categories are alphabet-nested (a/, b/, ...)" >&2
    echo "       and need flattening — see the header of this script." >&2
    exit 1
fi

echo "════════════════════════════════════════════════════════"
echo " Dataset : places365"
echo " Source  : $TRAIN_DIR ($N_CLASSES classes)"
echo " Latents : $LATENT_DIR"
echo " Packed  : $PACKED_DIR"
echo " GPUs    : $NUM_GPUS   batch/GPU: $BATCH_SIZE   image_size: $IMAGE_SIZE"
echo "════════════════════════════════════════════════════════"
df -h "$DATA_ROOT" | tail -1

# ── 1: encode ────────────────────────────────────────────────────────────────
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
NCCL_P2P_DISABLE=1 torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    encode_dataset.py \
    --data_path   "$TRAIN_DIR" \
    --output_path "$LATENT_DIR" \
    --image_size  $IMAGE_SIZE \
    --vae         $VAE \
    --batch_size  $BATCH_SIZE \
    --num_workers $NUM_WORKERS

# ── 2: pack ──────────────────────────────────────────────────────────────────
python pack_latents.py \
    --latent_path "$LATENT_DIR" \
    --output_path "$PACKED_DIR" \
    --num_workers $PACK_WORKERS

N_PACKED=$(find "$PACKED_DIR" -maxdepth 1 -name '*.npy' | wc -l)
echo "Done."
echo "  latents   : $LATENT_DIR"
echo "  packed    : $PACKED_DIR ($N_PACKED / $EXPECT_CLASSES .npy files)"
echo "  config    : set data.packed_latent_data_path: $PACKED_DIR"
echo "              set model.num_classes: $EXPECT_CLASSES"
echo "              set data.dataset_name: places365"
