"""
Decode packed VAE latents back into images (FID reference batch).

The subsets produced by build_variance_subset.py store latents that are already
multiplied by `latent_scale`, i.e.

    z_stored = latent_scale * (mean_base + std_base * eps) * 0.18215.

The VAE decoder expects the *unscaled* latent, so the reconstruction is

    x = decode( z_stored / (0.18215 * latent_scale) )
      = decode( mean_base + std_base * eps ).

That is the meaning of --decode_mode unscaled (the default) and it is the
reference batch that matches a sampler which divides its generated latents by
`latent_scale` before decoding (train.py / sample_ddp.py do this once
`latent_scale` is configured).  --decode_mode scaled skips the division and
therefore visualises what a scale-unaware sampler would produce; use it only if
the sampling side also ignores the scale.

Because the two variance subsets share the same base images, `unscaled` decoding
gives identical images for both — decode once and share the directory.

Usage (multi-GPU):
  torchrun --nproc_per_node=8 decode_latents.py \
      --packed_latent_path /scratch/.../variance_subsets/imagenet50k_std0.5/latents_packed/train \
      --output_path        /scratch/.../variance_subsets/reference_images/train

Usage (single GPU):
  python decode_latents.py --packed_latent_path ... --output_path ...
"""
import os
import json
import zlib
import argparse
from time import time

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from diffusers.models import AutoencoderKL


LATENT_SCALE = 0.18215
ORI_ORIGINAL, ORI_FLIPPED = 0, 1
PARAM_MEAN, PARAM_STD = 0, 1


def find_scale_info(packed_latent_path):
    """scale_info.json lives two levels above <root>/latents_packed/train."""
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(packed_latent_path)), "scale_info.json"),
        os.path.join(os.path.dirname(packed_latent_path), "scale_info.json"),
        os.path.join(packed_latent_path, "scale_info.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f), path
    return None, None


def find_manifest(packed_latent_path):
    """subset_manifest.json lives in the variance_subsets root (3 levels up)."""
    path = packed_latent_path
    for _ in range(4):
        path = os.path.dirname(path)
        candidate = os.path.join(path, "subset_manifest.json")
        if os.path.isfile(candidate):
            with open(candidate) as f:
                return json.load(f)
    return None


def build_npz_from_folder(image_root, npz_path, num=None):
    """ADM-style reference batch: a single .npz holding (N, H, W, 3) uint8."""
    files = []
    for root, _dirs, names in os.walk(image_root):
        for name in sorted(names):
            if name.endswith(".png"):
                files.append(os.path.join(root, name))
    files.sort()
    if num is not None:
        files = files[:num]
    samples = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz -> {npz_path} [shape={samples.shape}]")
    return npz_path


def main():
    parser = argparse.ArgumentParser(description="Decode packed VAE latents to images")
    parser.add_argument("--packed_latent_path", type=str, required=True,
                        help="Directory of packed per-class .npy latents")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output root; images are written to <root>/<class>/<name>.png")
    parser.add_argument("--latent_scale", type=float, default=None,
                        help="Scale baked into the stored latents. Default: read "
                             "from scale_info.json, else 1.0")
    parser.add_argument("--decode_mode", type=str, default="unscaled",
                        choices=["unscaled", "scaled"],
                        help="'unscaled' divides by latent_scale (true reconstruction); "
                             "'scaled' decodes the stored latent as-is")
    parser.add_argument("--posterior_mean", action="store_true",
                        help="Decode the posterior mean instead of a sample z ~ q(z|x)")
    parser.add_argument("--orientation", type=int, default=ORI_ORIGINAL, choices=[0, 1],
                        help="0 = original image, 1 = horizontally flipped")
    parser.add_argument("--vae", type=str, default="ema", choices=["ema", "mse"])
    parser.add_argument("--batch_size", type=int, default=32, help="Per-GPU batch size")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the posterior sampling noise (per-image, "
                             "so the output does not depend on the GPU count)")
    parser.add_argument("--make_npz", action="store_true",
                        help="Also build an ADM-style .npz reference batch (rank 0)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-decode images that already exist")
    args = parser.parse_args()

    # ── distributed setup (same convention as encode_dataset.py) ─────────────
    use_dist = "RANK" in os.environ
    if use_dist:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank, world_size = 0, 1
    device = rank % torch.cuda.device_count() if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # ── scale ────────────────────────────────────────────────────────────────
    info, info_path = find_scale_info(args.packed_latent_path)
    if args.latent_scale is not None:
        latent_scale = args.latent_scale
        scale_origin = "--latent_scale"
    elif info is not None:
        latent_scale = float(info["latent_scale"])
        scale_origin = info_path
    else:
        latent_scale = 1.0
        scale_origin = "default (no scale_info.json found)"
    divisor = LATENT_SCALE * latent_scale if args.decode_mode == "unscaled" else LATENT_SCALE

    manifest = find_manifest(args.packed_latent_path)

    npy_files = sorted(f for f in os.listdir(args.packed_latent_path) if f.endswith(".npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy latent files found in {args.packed_latent_path}")
    classes = [os.path.splitext(f)[0] for f in npy_files]

    if rank == 0:
        os.makedirs(args.output_path, exist_ok=True)
        print(f"Latents     : {args.packed_latent_path} ({len(classes)} classes)")
        print(f"Output      : {args.output_path}")
        print(f"latent_scale: {latent_scale:.8f}  (from {scale_origin})")
        print(f"decode_mode : {args.decode_mode}  ->  decode(z / {divisor:.8f})")
        print(f"latent src  : {'posterior mean' if args.posterior_mean else 'z ~ q(z|x)'}"
              f", orientation={args.orientation}")
    if use_dist:
        dist.barrier()

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    # Round-robin class sharding: each class file is read by exactly one rank.
    my_classes = classes[rank::world_size]
    n_written, n_skipped = 0, 0
    t0 = time()

    for cls in my_classes:
        arr = np.load(os.path.join(args.packed_latent_path, cls + ".npy"), mmap_mode="r")
        n_rows = int(arr.shape[0])

        names = None
        if manifest is not None and cls in manifest.get("classes", {}):
            files = manifest["classes"][cls].get("files")
            if files is not None and len(files) == n_rows:
                names = [os.path.splitext(f)[0] for f in files]
        if names is None:
            names = [f"{cls}_{i:05d}" for i in range(n_rows)]

        out_dir = os.path.join(args.output_path, cls)
        os.makedirs(out_dir, exist_ok=True)

        for start in range(0, n_rows, args.batch_size):
            stop = min(start + args.batch_size, n_rows)
            todo = [
                i for i in range(start, stop)
                if args.overwrite or not os.path.exists(
                    os.path.join(out_dir, names[i] + ".png"))
            ]
            n_skipped += (stop - start) - len(todo)
            if not todo:
                continue

            block = np.asarray(arr[todo], dtype=np.float32)   # (B, 2, 2, C, H, W)
            mean = torch.from_numpy(block[:, args.orientation, PARAM_MEAN]).to(device)
            if args.posterior_mean:
                latent = mean
            else:
                std = torch.from_numpy(block[:, args.orientation, PARAM_STD]).to(device)
                # Per-image seed so the noise is reproducible regardless of how
                # the work is sharded across GPUs. crc32 (not hash()) because
                # Python string hashing is randomised per process.
                cls_key = zlib.crc32(cls.encode()) & 0xFFFFFFFF
                noise = torch.stack([
                    torch.randn(
                        mean.shape[1:], dtype=mean.dtype,
                        generator=torch.Generator().manual_seed(
                            ((args.seed * 1_000_003 + cls_key) * 100_003 + i) % (2 ** 63 - 1)
                        ),
                    ) for i in todo
                ]).to(device)
                latent = mean + std * noise

            with torch.no_grad():
                # `latent` here is (mean + std*eps), i.e. z / 0.18215 with the
                # dataset scale still baked in; divisor removes both factors.
                images = vae.decode(latent * (LATENT_SCALE / divisor)).sample

            images = torch.clamp(127.5 * images + 128.0, 0, 255)
            images = images.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            for j, i in enumerate(todo):
                out_file = os.path.join(out_dir, names[i] + ".png")
                tmp = out_file + f".tmp.{os.getpid()}.png"
                Image.fromarray(images[j]).save(tmp)
                os.replace(tmp, out_file)
                n_written += 1

        del arr
        if rank == 0:
            done = n_written + n_skipped
            rate = done / (time() - t0 + 1e-6)
            print(f"\r[Rank 0] {done:,} images (new={n_written:,} skip={n_skipped:,}) "
                  f"{rate:.1f} img/s", end="", flush=True)

    if use_dist:
        dist.barrier()
    if rank == 0:
        print(f"\nRank 0 finished: {n_written:,} new, {n_skipped:,} skipped.")
        if args.make_npz:
            build_npz_from_folder(args.output_path, args.output_path.rstrip("/") + ".npz")
        print("Done.")
    if use_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
