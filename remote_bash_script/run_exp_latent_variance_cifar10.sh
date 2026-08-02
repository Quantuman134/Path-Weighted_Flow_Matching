#!/bin/bash
source /scratch/project/prj-02-visual-ai/hkzhang/miniconda3/etc/profile.d/conda.sh
conda activate SiT
cd /scratch/project/prj-02-visual-ai/hkzhang/Path-Weighted_Flow_Matching

# Single-GPU experiment: estimate sqrt((1/d) * tr(Cov(z))) of CIFAR-10 VAE latents.
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

DATA_PATH="/scratch/project/prj-02-visual-ai/hkzhang/cifar10/cifar10_imagefolder/train"
NUM_IMAGES=10000
IMAGE_SIZE=256          # must match the training pipeline (32x32 -> 256x256)
VAE="ema"               # 'ema' or 'mse' -- match the config you compare against
BATCH_SIZE=64
NUM_WORKERS=8
SEED=0
OUTPUT_DIR="./results/latent_variance"

echo "════════════════════════════════════════════════════════"
echo " Experiment : latent variance (CIFAR-10)"
echo " Data       : $DATA_PATH"
echo " Images     : $NUM_IMAGES   Image size : $IMAGE_SIZE"
echo " VAE        : sd-vae-ft-$VAE"
echo " Output     : $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════"

python exp_latent_variance.py \
    --data_path "${DATA_PATH}" \
    --num_images "${NUM_IMAGES}" \
    --image_size "${IMAGE_SIZE}" \
    --vae "${VAE}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --device "cuda:0" \
    --output_dir "${OUTPUT_DIR}" \
    --tag "cifar10" \
    --save_per_dim
