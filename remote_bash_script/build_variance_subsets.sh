#!/bin/bash
# Stage 1 of the latent-variance study.
#
#   1. sample 50,000 ImageNet images and slice their pre-encoded VAE latents
#   2. write two scaled copies with latent sigma 0.5 and 2.0 (source sigma 0.82)
#   3. decode the latents back to images -> FID reference batch
#
# Steps 1-2 need no GPU (the latents are already encoded); step 3 uses 8 GPUs.
set -e

source /scratch/project/prj-02-visual-ai/hkzhang/miniconda3/etc/profile.d/conda.sh
conda activate SiT
cd /scratch/project/prj-02-visual-ai/hkzhang/Path-Weighted_Flow_Matching

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=16

ILSVRC=/scratch/project/prj-02-visual-ai/hkzhang/ILSVRC
OUTPUT_ROOT=$ILSVRC/variance_subsets
NUM_IMAGES=50000
SOURCE_STD=0.82          # measured on 10k ImageNet latents (exp_latent_variance.py)
NUM_GPUS=8

# ── 1 + 2: sample, scale, write ──────────────────────────────────────────────
python build_variance_subset.py \
    --packed_latent_path $ILSVRC/latents_packed/train \
    --latent_path        $ILSVRC/latents/train \
    --image_path         $ILSVRC/Data/CLS-LOC/train \
    --output_root        $OUTPUT_ROOT \
    --source packed \
    --num_images $NUM_IMAGES \
    --sampling stratified \
    --target_std 0.5 2.0 \
    --source_std $SOURCE_STD \
    --name_prefix imagenet50k \
    --seed 0 \
    --verify

# ── 3: FID reference images, one set per subset ──────────────────────────────
# --decode_mode scaled leaves the dataset scale *in* the latent: the reference
# images live in the same (washed-out / saturated) pixel space that a sampler
# decoding with plain z/0.18215 produces. Each subset therefore gets its own
# reference set, and validation must decode with latent_scale = 1.0 to match
# (model_overrides.latent_scale: 1.0 in the validation configs).
for STD in 0.5 2.0; do
    MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
    NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
        decode_latents.py \
        --packed_latent_path $OUTPUT_ROOT/imagenet50k_std$STD/latents_packed/train \
        --output_path        $OUTPUT_ROOT/imagenet50k_std$STD/reference_images/train \
        --decode_mode scaled \
        --batch_size 32 \
        --seed 0
done

echo "Done."
echo "  latents   : $OUTPUT_ROOT/imagenet50k_std{0.5,2.0}/latents_packed/train"
echo "  FID refs  : $OUTPUT_ROOT/imagenet50k_std{0.5,2.0}/reference_images/train"
echo "  metadata  : $OUTPUT_ROOT/subset_manifest.json, */scale_info.json"
