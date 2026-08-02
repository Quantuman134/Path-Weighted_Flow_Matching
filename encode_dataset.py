"""
Pre-encode an ImageNet-format dataset to VAE latents for faster training.

Saves per-image .pt files mirroring the source directory structure.
Each file stores:
  {'mean': fp32(4,H,W), 'std': fp32(4,H,W),
   'mean_flip': fp32(4,H,W), 'std_flip': fp32(4,H,W)}
Both the original and horizontally-flipped image are encoded by the VAE so
that the flip augmentation during training is exact (not an approximation),
avoiding the small error from attention non-commutativity.

Usage (multi-GPU, recommended):
  torchrun --nproc_per_node=8 encode_dataset.py \
    --data_path /path/to/imagenet/train \
    --output_path /path/to/latents/train

Usage (single GPU):
  python encode_dataset.py \
    --data_path /path/to/imagenet/train \
    --output_path /path/to/latents/train
"""
import os
import argparse
from time import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from diffusers.models import AutoencoderKL
from PIL import Image


# ── helpers reused from train.py ──────────────────────────────────────────────

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class CenterCropTransform:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, pil_image):
        return center_crop_arr(pil_image, self.image_size)


class ImageFolderWithPaths(ImageFolder):
    """ImageFolder that also returns the source file path."""
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, y, self.samples[index][0]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-encode dataset to VAE latents")
    parser.add_argument("--data_path",   type=str, required=True,
                        help="Root of ImageFolder dataset to encode")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Root directory for output .pt latent files")
    parser.add_argument("--vae",         type=str, default="ema", choices=["ema", "mse"])
    parser.add_argument("--image_size",  type=int, default=256)
    parser.add_argument("--batch_size",  type=int, default=64,
                        help="Per-GPU batch size for encoding")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    # ── distributed setup ────────────────────────────────────────────────────
    use_dist = "RANK" in os.environ
    if use_dist:
        dist.init_process_group("nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank       = 0
        world_size = 1
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    # ── VAE ──────────────────────────────────────────────────────────────────
    if rank == 0:
        print(f"Loading VAE (sd-vae-ft-{args.vae})...", flush=True)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    # ── dataset (no random flip — applied to latents at training time) ───────
    transform = transforms.Compose([
        CenterCropTransform(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = ImageFolderWithPaths(args.data_path, transform=transform)
    if rank == 0:
        print(f"Dataset: {len(dataset):,} images in {len(dataset.classes)} classes")
        print(f"Output:  {args.output_path}")
        os.makedirs(args.output_path, exist_ok=True)

    # ── data loader ──────────────────────────────────────────────────────────
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    ) if use_dist else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ── encode loop ──────────────────────────────────────────────────────────
    rank0_total = len(sampler) if sampler is not None else len(dataset)
    n_new  = 0
    n_skip = 0
    t0     = time()

    for x, _y, paths in loader:
        # Determine output paths and which images still need encoding
        out_paths   = []
        todo_mask   = []
        for src in paths:
            rel     = os.path.relpath(src, args.data_path)
            out     = os.path.join(args.output_path,
                                   os.path.splitext(rel)[0] + ".pt")
            out_paths.append(out)
            exists  = os.path.exists(out)
            todo_mask.append(not exists)
            if exists:
                n_skip += 1

        todo_indices = [i for i, m in enumerate(todo_mask) if m]

        if todo_indices:
            x_todo = x[todo_indices].to(device)
            with torch.no_grad():
                # Encode original and horizontally-flipped images separately so
                # the flip augmentation during training is exact (no attention error).
                posterior      = vae.encode(x_todo).latent_dist
                means          = posterior.mean.cpu()           # fp32 (B, 4, H, W)
                stds           = posterior.std.cpu()
                posterior_flip = vae.encode(torch.flip(x_todo, dims=[-1])).latent_dist
                means_flip     = posterior_flip.mean.cpu()
                stds_flip      = posterior_flip.std.cpu()

            for j, i in enumerate(todo_indices):
                out = out_paths[i]
                os.makedirs(os.path.dirname(out), exist_ok=True)
                # Rank-unique temp name: DistributedSampler pads the index list
                # so the last ranks re-encode a few of rank 0's images. A shared
                # temp path would let two ranks interleave writes into one file.
                tmp = f"{out}.tmp.{rank}"
                torch.save({
                    "mean":      means[j],
                    "std":       stds[j],
                    "mean_flip": means_flip[j],
                    "std_flip":  stds_flip[j],
                }, tmp)
                os.replace(tmp, out)  # atomic — no partial files on crash
                n_new += 1

        if rank == 0:
            done    = n_new + n_skip
            elapsed = time() - t0 + 1e-6
            rate    = done / elapsed
            eta_min = (rank0_total - done) / max(rate, 1) / 60
            print(f"\r[Rank 0] {done:,}/{rank0_total:,} (this rank)  "
                  f"new={n_new}  skip={n_skip}  "
                  f"{rate:.0f} img/s  ETA {eta_min:.1f} min",
                  end="", flush=True)

    if use_dist:
        dist.barrier()
    if rank == 0:
        print(f"\nFinished. Encoded {n_new:,} new images, skipped {n_skip:,} existing.")


if __name__ == "__main__":
    main()
