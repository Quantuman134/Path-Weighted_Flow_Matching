"""
Pack per-file VAE latents into per-class .npy files for faster training.

Input:  latents/train/n01440764/img1.pt
        {'mean': fp32(4,H,W), 'std': fp32(4,H,W),
         'mean_flip': fp32(4,H,W), 'std_flip': fp32(4,H,W)}

Output: latents_packed/train/n01440764.npy  shape (N, 2, 2, 4, H, W) float32
        axis 1: orientation — 0=original, 1=flipped
        axis 2: parameter  — 0=mean,      1=std

This reduces 1.28M individual file-open operations to 1000, which eliminates
the NFS metadata bottleneck. The original per-file latents are left untouched.

Usage:
  python pack_latents.py \
    --latent_path  /scratch/.../ILSVRC/latents/train \
    --output_path  /scratch/.../ILSVRC/latents_packed/train \
    --num_workers  32
"""
import os
import argparse
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time


def pack_class(class_dir, class_name, output_path):
    """Read all .pt files in class_dir and save as one .npy file."""
    out_file = os.path.join(output_path, class_name + ".npy")
    if os.path.exists(out_file):
        return 0  # already done

    pt_files = sorted(f for f in os.listdir(class_dir) if f.endswith(".pt"))
    if not pt_files:
        return 0

    samples = []
    for fname in pt_files:
        data = torch.load(os.path.join(class_dir, fname), weights_only=True)
        orig    = np.stack([data["mean"].numpy(),      data["std"].numpy()])       # (2, 4, H, W)
        flipped = np.stack([data["mean_flip"].numpy(), data["std_flip"].numpy()])  # (2, 4, H, W)
        samples.append(np.stack([orig, flipped]))  # (2, 2, 4, H, W)

    arr = np.stack(samples)  # (N, 2, 2, 4, H, W) float32

    tmp = out_file[:-4] + f".tmp.{os.getpid()}.npy"  # must end in .npy so np.save won't append it again
    np.save(tmp, arr)
    os.replace(tmp, out_file)
    return len(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_path",  type=str, required=True,
                        help="Directory of per-file .pt latents (output of encode_dataset.py)")
    parser.add_argument("--output_path",  type=str, required=True,
                        help="Output directory for packed .npy files")
    parser.add_argument("--num_workers",  type=int, default=32,
                        help="Parallel threads (one per class)")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    classes = sorted(
        d for d in os.listdir(args.latent_path)
        if os.path.isdir(os.path.join(args.latent_path, d))
    )
    print(f"Found {len(classes)} classes  →  {args.output_path}")

    n_samples = 0
    n_skipped = 0
    t0 = time()

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {
            pool.submit(
                pack_class,
                os.path.join(args.latent_path, cls),
                cls,
                args.output_path,
            ): cls
            for cls in classes
        }
        for i, fut in enumerate(as_completed(futures)):
            n = fut.result()
            if n:
                n_samples += n
            else:
                n_skipped += 1
            if (i + 1) % 20 == 0 or (i + 1) == len(classes):
                elapsed = time() - t0 + 1e-6
                eta = (len(classes) - i - 1) / ((i + 1) / elapsed)
                print(f"\r[{i+1:4d}/{len(classes)}] classes  "
                      f"{n_samples:,} samples packed  "
                      f"{n_skipped} skipped  ETA {eta/60:.1f} min",
                      end="", flush=True)

    print(f"\nDone. {n_samples:,} samples in {len(classes) - n_skipped} classes.")


if __name__ == "__main__":
    main()
