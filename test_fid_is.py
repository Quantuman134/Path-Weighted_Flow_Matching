"""
test_fid_is.py — Sanity-check FID and IS computation.

Two tests:
  1. IS on 50K ImageNet val images   → expect ~233 (high, real diverse images)
  2. FID between 50K val images and 50K train images → expect ~0 (both real ImageNet)

Usage:
    python test_fid_is.py

Paths are hard-coded to the known ImageNet location on this machine.
"""

import sys
import os
import numpy as np
import torch
from torchvision import datasets, models as tv_models, transforms

_SIT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SIT_DIR not in sys.path:
    sys.path.insert(0, _SIT_DIR)

from FID import get_inception_features
from IS import get_inception_probs, compute_is_from_probs
from exp_model_validation import compute_fid_from_features, extract_features_chunked

# ── Config ────────────────────────────────────────────────────────────────────

VAL_PATH   = "/scratch/project/prj-02-visual-ai/hkzhang/ILSVRC/Data/CLS-LOC/val"
TRAIN_PATH = "/scratch/project/prj-02-visual-ai/hkzhang/ILSVRC/Data/CLS-LOC/train"
IMAGE_SIZE = 256
NUM_IMAGES = 50000
BATCH_SIZE = 256
CHUNK_SIZE = 5000
DEVICE     = "cuda:0"
SEED       = 42


# ── Data loading ──────────────────────────────────────────────────────────────

def load_images_as_uint8(root: str, num: int, image_size: int,
                          batch_size: int, seed: int) -> np.ndarray:
    """Load `num` images from an ImageFolder dataset; return uint8 (N,3,H,W)."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root, transform=transform)
    rng     = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(num, len(dataset)), replace=False)
    subset  = torch.utils.data.Subset(dataset, indices.tolist())
    loader  = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    batches = []
    total   = 0
    for x, _ in loader:
        uint8 = (x * 255).clamp(0, 255).to(torch.uint8).numpy()
        batches.append(uint8)
        total += len(uint8)
        print(f"\r  Loaded {total}/{len(subset)} images", end="", flush=True)
    print()
    return np.concatenate(batches, axis=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.set_grad_enabled(False)

    # ── Pre-load Inception v3 for IS (fc layer intact) ────────────────────────
    print("Pre-loading Inception v3 for IS...")
    inception_for_is = tv_models.inception_v3(
        weights=tv_models.Inception_V3_Weights.DEFAULT
    )
    inception_for_is.aux_logits = False
    inception_for_is.eval().to(DEVICE)

    # ════════════════════════════════════════════════════════════════════════════
    # Test 1: IS on 50K val images
    # Expected: ~233 (real ImageNet images are high quality and diverse)
    # ════════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*55}")
    print(f" Test 1: IS on {NUM_IMAGES:,} val images")
    print(f"{'═'*55}")

    print(f"Loading val images from {VAL_PATH} ...")
    val_uint8 = load_images_as_uint8(VAL_PATH, NUM_IMAGES, IMAGE_SIZE, BATCH_SIZE, SEED)
    print(f"Loaded {len(val_uint8):,} val images — shape: {val_uint8.shape}")

    print("Computing IS probs (chunked)...")
    val_probs = extract_features_chunked(
        val_uint8, BATCH_SIZE, CHUNK_SIZE, DEVICE,
        inception_model=inception_for_is, mode="is",
    )
    is_mean, is_std = compute_is_from_probs(val_probs, splits=10)
    print(f"\n  IS (val 50K) = {is_mean:.4f} ± {is_std:.4f}")
    print(f"  Expected:      ~233 ± small  (standard ImageNet IS)")

    # ════════════════════════════════════════════════════════════════════════════
    # Test 2: FID between 50K val and 50K train images
    # Expected: ~0 (both are real ImageNet images, same distribution)
    # ════════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*55}")
    print(f" Test 2: FID — val vs train ({NUM_IMAGES:,} images each)")
    print(f"{'═'*55}")

    print("Extracting FID features for val images (chunked)...")
    val_features = extract_features_chunked(
        val_uint8, BATCH_SIZE, CHUNK_SIZE, DEVICE, mode="fid",
    )
    del val_uint8

    print(f"\nLoading train images from {TRAIN_PATH} ...")
    train_uint8 = load_images_as_uint8(
        TRAIN_PATH, NUM_IMAGES, IMAGE_SIZE, BATCH_SIZE, seed=SEED + 1
    )
    print(f"Loaded {len(train_uint8):,} train images — shape: {train_uint8.shape}")

    print("Extracting FID features for train images (chunked)...")
    train_features = extract_features_chunked(
        train_uint8, BATCH_SIZE, CHUNK_SIZE, DEVICE, mode="fid",
    )
    del train_uint8

    fid = compute_fid_from_features(val_features, train_features)
    print(f"\n  FID (val vs train) = {fid:.4f}")
    print(f"  Expected:            ~0 (both real ImageNet, same distribution)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f" Summary")
    print(f"{'─'*55}")
    print(f"  IS  (val 50K)       = {is_mean:.4f} ± {is_std:.4f}")
    print(f"  FID (val vs train)  = {fid:.4f}")
    print(f"{'─'*55}")


if __name__ == "__main__":
    main()
