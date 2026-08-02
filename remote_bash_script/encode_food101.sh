#!/bin/bash
# Pre-encode Food-101 to VAE latents, then pack to per-class .npy.
#
#   1. encode  train/<class>/*.jpg -> latents/train/<class>/*.pt   (8 GPUs)
#   2. pack    latents/train       -> latents_packed/train/*.npy   (CPU)
#
# Expects the dataset already downloaded and split into ImageFolder layout:
#   wget -c http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
# then split images/ into train/ and test/ using food-101/meta/{train,test}.txt.
#
# Both stages are resumable: encode_dataset.py skips existing .pt files and
# pack_latents.py skips existing .npy files. Re-run after an interruption.
set -e

source /scratch/project/prj-02-visual-ai/hkzhang/miniconda3/etc/profile.d/conda.sh
conda activate SiT
cd /scratch/project/prj-02-visual-ai/hkzhang/Path-Weighted_Flow_Matching

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=16

DATA_ROOT=/scratch/project/prj-02-visual-ai/hkzhang/food101
TRAIN_DIR=$DATA_ROOT/train
LATENT_DIR=$DATA_ROOT/latents/train
PACKED_DIR=$DATA_ROOT/latents_packed/train

NUM_GPUS=8
IMAGE_SIZE=256
VAE=ema                  # matches model.vae in the config
BATCH_SIZE=64            # per-GPU
NUM_WORKERS=8            # per-GPU dataloader workers
PACK_WORKERS=32

EXPECT_CLASSES=101       # Food-101: 101 classes, 75,750 train images

# ── preconditions ────────────────────────────────────────────────────────────
if [ ! -d "$TRAIN_DIR" ]; then
    echo "ERROR: $TRAIN_DIR not found — download and split Food-101 first." >&2
    exit 1
fi

N_CLASSES=$(find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
if [ "$N_CLASSES" -ne "$EXPECT_CLASSES" ]; then
    echo "ERROR: found $N_CLASSES class dirs in $TRAIN_DIR, expected $EXPECT_CLASSES." >&2
    echo "       ImageFolder infers num_classes from this — fix the layout first." >&2
    exit 1
fi

echo "════════════════════════════════════════════════════════"
echo " Dataset : food101"
echo " Source  : $TRAIN_DIR ($N_CLASSES classes)"
echo " Latents : $LATENT_DIR"
echo " Packed  : $PACKED_DIR"
echo " GPUs    : $NUM_GPUS   batch/GPU: $BATCH_SIZE   image_size: $IMAGE_SIZE"
echo "════════════════════════════════════════════════════════"

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
echo "              set data.dataset_name: food101"
