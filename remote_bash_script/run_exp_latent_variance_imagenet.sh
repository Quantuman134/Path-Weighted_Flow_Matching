#!/bin/bash
source /scratch/project/prj-02-visual-ai/hkzhang/miniconda3/etc/profile.d/conda.sh
conda activate SiT
cd /scratch/project/prj-02-visual-ai/hkzhang/Path-Weighted_Flow_Matching

# Single-GPU experiment: estimate sqrt((1/d) * tr(Cov(z))) of ImageNet VAE latents.
# Reads the pre-encoded packed latents (no VAE forward pass, no image decoding),
# which are exactly the latents train.py consumes for ImageNet runs.
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

PACKED_LATENT_PATH="/scratch/project/prj-02-visual-ai/hkzhang/ILSVRC/latents_packed/train"
NUM_IMAGES=10000
VAE="ema"               # the VAE encode_dataset.py used to build the latents
BATCH_SIZE=64
SEED=0
ORIENTATION=0           # 0 = original image, 1 = horizontally flipped
OUTPUT_DIR="./results/latent_variance"

echo "════════════════════════════════════════════════════════"
echo " Experiment : latent variance (ImageNet)"
echo " Latents    : $PACKED_LATENT_PATH"
echo " Images     : $NUM_IMAGES"
echo " Output     : $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════"

python exp_latent_variance.py \
    --packed_latent_path "${PACKED_LATENT_PATH}" \
    --num_images "${NUM_IMAGES}" \
    --vae "${VAE}" \
    --batch_size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --orientation "${ORIENTATION}" \
    --device "cuda:0" \
    --output_dir "${OUTPUT_DIR}" \
    --tag "imagenet" \
    --save_per_dim

# ── Alternative: encode raw ImageNet images with the VAE on the fly ──────────
# Slower (~5000 VAE forward passes) but does not depend on the packed latents.
#
# python exp_latent_variance.py \
#     --data_path /scratch/project/prj-02-visual-ai/hkzhang/ILSVRC/Data/CLS-LOC/train \
#     --num_images 10000 --image_size 256 --vae ema \
#     --batch_size 64 --num_workers 8 --seed 0 \
#     --device "cuda:0" --output_dir "${OUTPUT_DIR}" --tag "imagenet_raw" --save_per_dim
