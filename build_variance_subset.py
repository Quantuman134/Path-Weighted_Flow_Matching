"""
Build fixed-size ImageNet subsets whose VAE latents have a prescribed std.

Motivation
----------
The training latent  z = (mean + std * eps) * 0.18215  has an empirical
    sigma = sqrt( (1/d) tr(Cov(z)) ) ~= 0.82
on ImageNet (see exp_latent_variance.py).  To study how the path-weighted
flow-matching weightings behave as a function of the *data* variance we need
datasets that are identical except for that scale.

Because scaling is linear, multiplying the whole latent by a constant s
    s * z = (s * mean + s * std * eps) * 0.18215
is *exactly* equivalent to storing (s * mean, s * std) in the packed posterior
file.  No re-encoding is needed and the scaled sigma is exactly s * sigma.

Pipeline
--------
1. Randomly sample N images (default 50,000) from the source dataset.
2. Obtain their latent posteriors (mean, std).  Two sources are supported:
     - 'packed'  : slice the rows out of the pre-encoded per-class .npy files
                   produced by encode_dataset.py + pack_latents.py.  This is
                   bit-identical to re-encoding and needs no GPU.
     - 'images'  : run the VAE on the sampled images (use only when the packed
                   latents do not exist; see --source images).
3. Measure sigma of the sampled subset (streaming float64 moments).
4. For every requested target std, write a scaled copy of the packed latents
   with  s = target_std / source_std.
5. Write a manifest (which images were picked) and a scale_info.json per subset
   so the downstream training / decoding scripts can recover the scale.

Output layout
-------------
  <output_root>/
      subset_manifest.json                     # sampled images, seed, counts
      base_stats.json                          # sigma of the unscaled subset
      imagenet50k_std0.5/
          scale_info.json
          latents_packed/train/<class>.npy     # (N, 2, 2, 4, H, W) float32
      imagenet50k_std2.0/
          ...

Usage (slice from the existing packed ImageNet latents -- recommended):
  python build_variance_subset.py \
      --packed_latent_path /scratch/.../ILSVRC/latents_packed/train \
      --image_path         /scratch/.../ILSVRC/Data/CLS-LOC/train \
      --output_root        /scratch/.../ILSVRC/variance_subsets \
      --num_images 50000 --target_std 0.5 2.0 --source_std 0.82

Usage (encode from raw images, single or multi GPU via torchrun):
  python build_variance_subset.py --source images \
      --image_path  /scratch/.../ILSVRC/Data/CLS-LOC/train \
      --output_root /scratch/.../ILSVRC/variance_subsets \
      --num_images 50000 --target_std 0.5 2.0
"""
import os
import json
import bisect
import argparse
from time import time

import numpy as np
import torch


# SD-VAE latent scaling factor used everywhere in this repo (train.py, sample.py).
LATENT_SCALE = 0.18215

# Packed latent axis layout: (N, orientation, parameter, C, H, W)
ORI_ORIGINAL, ORI_FLIPPED = 0, 1
PARAM_MEAN, PARAM_STD = 0, 1


def fmt_std(value):
    """Compact, stable string for a target std: 0.5 -> '0.5', 2.0 -> '2.0'.

    Used for directory and dataset names, so it must match the tags hard-coded in
    configs/sit_config_B_*_imagenet50k_std*.yaml.
    """
    s = f"{value:g}"
    return s if "." in s else s + ".0"


# ── subset selection ─────────────────────────────────────────────────────────

def scan_packed_classes(packed_root):
    """Returns (class_names, row_counts, latent_shape) for a packed latent dir."""
    npy_files = sorted(f for f in os.listdir(packed_root) if f.endswith(".npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy latent files found in {packed_root}")

    classes, counts = [], []
    latent_shape = None
    for fname in npy_files:
        arr = np.load(os.path.join(packed_root, fname), mmap_mode="r")
        if arr.ndim != 6:
            raise ValueError(
                f"{fname}: expected a packed array of shape (N, 2, 2, C, H, W), "
                f"got {arr.shape}"
            )
        if latent_shape is None:
            latent_shape = tuple(arr.shape[3:])          # (C, H, W)
        elif tuple(arr.shape[3:]) != latent_shape:
            raise ValueError(
                f"{fname} has latent shape {tuple(arr.shape[3:])}, "
                f"expected {latent_shape}"
            )
        classes.append(os.path.splitext(fname)[0])
        counts.append(int(arr.shape[0]))
        del arr                                          # release the mmap
    return classes, counts, latent_shape


def select_uniform(counts, num_images, rng):
    """Uniform sample without replacement over the flat index space.

    Returns a list (one entry per class) of sorted local row indices.  Sorting
    keeps the memory-map reads sequential; the sample itself is still uniform.
    """
    cum = np.cumsum(counts)
    total = int(cum[-1])
    n_take = min(num_images, total)
    flat = rng.choice(total, size=n_take, replace=False)
    flat.sort()

    per_class = [[] for _ in counts]
    for idx in flat:
        pos = int(np.searchsorted(cum, idx, side="right"))
        local = int(idx) - (int(cum[pos - 1]) if pos > 0 else 0)
        per_class[pos].append(local)
    return [np.array(sorted(rows), dtype=np.int64) for rows in per_class]


def select_stratified(counts, num_images, rng):
    """Class-balanced sample: floor(N/C) rows per class, remainder spread over
    randomly chosen classes.  Classes with too few rows contribute all of theirs.
    """
    n_classes = len(counts)
    base = num_images // n_classes
    remainder = num_images - base * n_classes
    bonus = set(rng.choice(n_classes, size=remainder, replace=False).tolist()) \
        if remainder else set()

    per_class = []
    for i, n_avail in enumerate(counts):
        want = base + (1 if i in bonus else 0)
        take = min(want, n_avail)
        rows = rng.choice(n_avail, size=take, replace=False)
        per_class.append(np.array(sorted(rows.tolist()), dtype=np.int64))
    return per_class


def source_filenames(class_name, n_rows, latent_path, image_path):
    """Recover the source file names for the rows of one packed class file.

    pack_latents.py stores rows in sorted .pt-filename order, and encode_dataset.py
    mirrors the image file names, so the sorted listing of either directory gives
    the row order.  The per-file latent directory is preferred (it is what was
    actually packed); the image directory is the fallback.  Returns None when
    neither listing matches the row count (the row index is then authoritative).
    """
    for root, ext in ((latent_path, ".pt"), (image_path, None)):
        if root is None:
            continue
        cls_dir = os.path.join(root, class_name)
        if not os.path.isdir(cls_dir):
            continue
        names = sorted(
            f for f in os.listdir(cls_dir)
            if ext is None or f.endswith(ext)
        )
        if len(names) == n_rows:
            return names
    return None


# ── streaming moments ────────────────────────────────────────────────────────

class MomentAccumulator:
    """Per-dimension sum(z) and sum(z^2) in float64 (same estimator as
    exp_latent_variance.py, so the numbers are directly comparable)."""

    def __init__(self, dim, device):
        self.dim = int(dim)
        self.n = 0
        self.sum = torch.zeros(self.dim, dtype=torch.float64, device=device)
        self.sum_sq = torch.zeros(self.dim, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, z):
        zf = z.reshape(z.shape[0], -1).to(torch.float64)
        self.n += int(zf.shape[0])
        self.sum += zf.sum(dim=0)
        self.sum_sq += (zf * zf).sum(dim=0)

    def sigma(self):
        """sqrt( (1/d) tr(Cov(z)) ) -- the RMS per-coordinate std."""
        assert self.n > 1, "Need at least 2 samples to estimate a variance."
        mean = self.sum / self.n
        sq_dev = (self.sum_sq - self.n * mean * mean).clamp_min(0.0)
        var = sq_dev / (self.n - 1)
        return float(torch.sqrt(var.mean()).item())

    def summary(self):
        mean = self.sum / self.n
        sq_dev = (self.sum_sq - self.n * mean * mean).clamp_min(0.0)
        var = sq_dev / (self.n - 1)
        return {
            "n": self.n,
            "d": self.dim,
            "sigma": float(torch.sqrt(var.mean()).item()),
            "mean_variance_per_dim": float(var.mean().item()),
            "global_mean": float(mean.mean().item()),
            "E_z_sq_per_dim": float(self.sum_sq.sum().item() / (self.n * self.dim)),
        }


def iter_selected_blocks(packed_root, classes, selection, chunk_rows=256):
    """Yields (class_name, rows, block) where block is a float32 ndarray of shape
    (len(rows), 2, 2, C, H, W) read from the packed file."""
    for cls, rows in zip(classes, selection):
        if len(rows) == 0:
            continue
        arr = np.load(os.path.join(packed_root, cls + ".npy"), mmap_mode="r")
        for start in range(0, len(rows), chunk_rows):
            chunk = rows[start:start + chunk_rows]
            yield cls, chunk, np.asarray(arr[chunk])
        del arr


def measure_sigma(packed_root, classes, selection, latent_shape, device,
                  orientation, seed, log_every=50):
    """Streaming estimate of sigma over the selected rows."""
    d = int(np.prod(latent_shape))
    acc = MomentAccumulator(d, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    orientations = [ORI_ORIGINAL, ORI_FLIPPED] if orientation == "both" else [int(orientation)]

    total = int(sum(len(r) for r in selection))
    done = 0
    t0 = time()
    with torch.no_grad():
        for _cls, chunk, block in iter_selected_blocks(packed_root, classes, selection):
            for ori in orientations:
                mean = torch.from_numpy(block[:, ori, PARAM_MEAN]).to(device)
                std = torch.from_numpy(block[:, ori, PARAM_STD]).to(device)
                noise = torch.randn(mean.shape, generator=generator,
                                    device=device, dtype=mean.dtype)
                acc.update((mean + std * noise) * LATENT_SCALE)
            done += len(chunk)
            if done % log_every == 0 or done == total:
                rate = done / (time() - t0 + 1e-6)
                print(f"\r  measuring sigma: {done:,}/{total:,} ({rate:.0f} lat/s)",
                      end="", flush=True)
    print()
    return acc


# ── VAE encoding fallback (--source images) ──────────────────────────────────

def encode_subset_to_packed(image_root, classes, selection, filenames_per_class,
                            out_packed_root, image_size, vae_name, device,
                            batch_size):
    """Encode the sampled images with the VAE and write packed per-class .npy
    files in exactly the format pack_latents.py produces."""
    from PIL import Image
    from torchvision import transforms
    from diffusers.models import AutoencoderKL
    from encode_dataset import CenterCropTransform

    os.makedirs(out_packed_root, exist_ok=True)
    transform = transforms.Compose([
        CenterCropTransform(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    print(f"Loading VAE (sd-vae-ft-{vae_name}) ...", flush=True)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_name}").to(device)
    vae.eval()

    total = int(sum(len(r) for r in selection))
    done = 0
    t0 = time()
    for cls, rows, names in zip(classes, selection, filenames_per_class):
        if len(rows) == 0:
            continue
        out_file = os.path.join(out_packed_root, cls + ".npy")
        if os.path.exists(out_file):
            done += len(rows)
            continue
        if names is None:
            raise RuntimeError(
                f"Cannot encode class {cls}: source file names are unknown. "
                "Pass --image_path pointing at the ImageFolder root."
            )
        picked = [names[i] for i in rows]
        samples = []
        for start in range(0, len(picked), batch_size):
            batch_names = picked[start:start + batch_size]
            imgs = torch.stack([
                transform(Image.open(os.path.join(image_root, cls, fn)).convert("RGB"))
                for fn in batch_names
            ]).to(device)
            with torch.no_grad():
                post = vae.encode(imgs).latent_dist
                post_flip = vae.encode(torch.flip(imgs, dims=[-1])).latent_dist
                orig = torch.stack([post.mean, post.std], dim=1).cpu().numpy()
                flip = torch.stack([post_flip.mean, post_flip.std], dim=1).cpu().numpy()
            # (B, 2, 2, C, H, W): axis 1 = orientation, axis 2 = (mean, std)
            samples.append(np.stack([orig, flip], axis=1).astype(np.float32))
            done += len(batch_names)
            rate = done / (time() - t0 + 1e-6)
            print(f"\r  encoding: {done:,}/{total:,} ({rate:.1f} img/s)",
                  end="", flush=True)
        arr = np.concatenate(samples, axis=0)
        tmp = out_file[:-4] + f".tmp.{os.getpid()}.npy"
        np.save(tmp, arr)
        os.replace(tmp, out_file)
    print()
    return out_packed_root


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build ImageNet subsets with a prescribed latent std"
    )
    parser.add_argument("--packed_latent_path", type=str,
                        default="/scratch/project/prj-02-visual-ai/hkzhang/ILSVRC/latents_packed/train",
                        help="Source packed latents (<class>.npy) to slice from")
    parser.add_argument("--latent_path", type=str,
                        default="/scratch/project/prj-02-visual-ai/hkzhang/ILSVRC/latents/train",
                        help="Per-file .pt latents; used only to recover row -> file names")
    parser.add_argument("--image_path", type=str,
                        default="/scratch/project/prj-02-visual-ai/hkzhang/ILSVRC/Data/CLS-LOC/train",
                        help="ImageFolder root of the source images (manifest / --source images)")
    parser.add_argument("--output_root", type=str,
                        default="/scratch/project/prj-02-visual-ai/hkzhang/ILSVRC/variance_subsets",
                        help="Where the scaled subsets are written")
    parser.add_argument("--source", type=str, default="packed", choices=["packed", "images"],
                        help="'packed' slices pre-encoded latents (no GPU); "
                             "'images' runs the VAE on the sampled images")
    parser.add_argument("--num_images", type=int, default=50000)
    parser.add_argument("--sampling", type=str, default="stratified",
                        choices=["stratified", "uniform"],
                        help="'stratified' = equal number of images per class "
                             "(50/class at N=50k); 'uniform' = uniform over all images")
    parser.add_argument("--target_std", type=float, nargs="+", default=[0.5, 2.0],
                        help="Target latent sigma for each subset")
    parser.add_argument("--source_std", type=str, default="0.82",
                        help="Reference sigma of the unscaled latents used to derive "
                             "the scale (s = target/source). 'auto' uses the sigma "
                             "measured on this subset instead.")
    parser.add_argument("--name_prefix", type=str, default="imagenet50k",
                        help="Subset directory / dataset name prefix")
    parser.add_argument("--orientation", type=str, default="0", choices=["0", "1", "both"],
                        help="Orientation used for the sigma measurement "
                             "(0 = original image, matching exp_latent_variance.py)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for --source images encoding")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size for --source images encoding")
    parser.add_argument("--vae", type=str, default="ema", choices=["ema", "mse"])
    parser.add_argument("--save_base", action="store_true",
                        help="Also write the unscaled (s=1) copy of the subset")
    parser.add_argument("--verify", action="store_true",
                        help="Re-measure sigma of every written subset (extra I/O)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_root, exist_ok=True)

    # ── 1. pick the images ───────────────────────────────────────────────────
    if not os.path.isdir(args.packed_latent_path):
        raise FileNotFoundError(
            f"packed_latent_path does not exist: {args.packed_latent_path}"
        )
    classes, counts, latent_shape = scan_packed_classes(args.packed_latent_path)
    total_available = int(sum(counts))
    print(f"Source  : {args.packed_latent_path}")
    print(f"          {total_available:,} latents in {len(classes)} classes, "
          f"latent shape {latent_shape}")

    if args.num_images > total_available:
        raise ValueError(
            f"Requested {args.num_images:,} images but the source has only "
            f"{total_available:,}."
        )

    if args.sampling == "stratified":
        selection = select_stratified(counts, args.num_images, rng)
    else:
        selection = select_uniform(counts, args.num_images, rng)
    n_selected = int(sum(len(r) for r in selection))
    n_empty = sum(1 for r in selection if len(r) == 0)
    print(f"Sampled : {n_selected:,} images ({args.sampling}, seed={args.seed})")
    if n_selected != args.num_images:
        print(f"[warn] selected {n_selected:,} rows, requested {args.num_images:,} "
              f"(some classes ran out of images).")
    if n_empty:
        print(f"[warn] {n_empty} classes received no image. The class -> label "
              f"mapping of the subset will differ from the full ImageNet ordering.")

    # Row -> source file name, so the subset is reproducible and traceable.
    filenames_per_class = [
        source_filenames(cls, n, args.latent_path, args.image_path)
        for cls, n in zip(classes, counts)
    ]
    n_unresolved = sum(
        1 for names, rows in zip(filenames_per_class, selection)
        if names is None and len(rows) > 0
    )
    if n_unresolved:
        print(f"[warn] could not resolve source file names for {n_unresolved} "
              f"classes; the manifest stores row indices only.")

    # ── 2. latents of the subset ─────────────────────────────────────────────
    if args.source == "images":
        base_packed = os.path.join(args.output_root, "base", "latents_packed", "train")
        print(f"Encoding the sampled images with the VAE -> {base_packed}")
        encode_subset_to_packed(
            args.image_path, classes, selection, filenames_per_class,
            base_packed, args.image_size, args.vae, device, args.batch_size,
        )
        # From here on the freshly written files are the source, and every row
        # in them is selected.
        src_packed = base_packed
        src_classes, src_counts, latent_shape = scan_packed_classes(src_packed)
        src_selection = [np.arange(n, dtype=np.int64) for n in src_counts]
    else:
        src_packed = args.packed_latent_path
        src_classes = classes
        src_selection = selection

    # ── 3. measure sigma of the unscaled subset ──────────────────────────────
    print("Measuring sigma of the unscaled subset ...")
    acc = measure_sigma(src_packed, src_classes, src_selection, latent_shape,
                        device, args.orientation, args.seed)
    base_stats = acc.summary()
    sigma_measured = base_stats["sigma"]
    print(f"  sigma (measured, orientation={args.orientation}) = {sigma_measured:.6f}")

    if args.source_std.lower() == "auto":
        source_std = sigma_measured
        source_std_origin = "measured on this subset"
    else:
        source_std = float(args.source_std)
        source_std_origin = "user supplied (--source_std)"
        rel = abs(sigma_measured - source_std) / source_std
        if rel > 0.05:
            print(f"[warn] --source_std={source_std} differs from the measured "
                  f"sigma {sigma_measured:.4f} by {rel * 100:.1f}%. The achieved "
                  f"std will be target * {sigma_measured / source_std:.4f}.")
    print(f"  reference sigma used for the scale = {source_std:.6f} "
          f"({source_std_origin})")

    with open(os.path.join(args.output_root, "base_stats.json"), "w") as f:
        json.dump({
            "source": src_packed,
            "sampling": args.sampling,
            "num_images": n_selected,
            "seed": args.seed,
            "orientation": args.orientation,
            "latent_scale_constant": LATENT_SCALE,
            "stats": base_stats,
        }, f, indent=2)

    # ── 4. write the scaled subsets ──────────────────────────────────────────
    targets = list(args.target_std)
    scales = {t: t / source_std for t in targets}
    if args.save_base:
        scales[None] = 1.0

    def subset_dir(target):
        tag = "base" if target is None else f"std{fmt_std(target)}"
        return os.path.join(args.output_root, f"{args.name_prefix}_{tag}")

    out_dirs = {t: os.path.join(subset_dir(t), "latents_packed", "train") for t in scales}
    for path in out_dirs.values():
        os.makedirs(path, exist_ok=True)

    for t, s in scales.items():
        label = "base (unscaled)" if t is None else f"target std {t:g}"
        print(f"Writing {label}: scale = {s:.8f} -> {out_dirs[t]}")

    written = {t: 0 for t in scales}
    t0 = time()
    total_rows = int(sum(len(r) for r in src_selection))
    done = 0
    for cls, rows in zip(src_classes, src_selection):
        if len(rows) == 0:
            continue
        arr = np.load(os.path.join(src_packed, cls + ".npy"), mmap_mode="r")
        block = np.asarray(arr[rows], dtype=np.float32)   # (K, 2, 2, C, H, W)
        del arr
        for t, s in scales.items():
            # Scaling mean and std by the same factor scales the sampled latent
            # z = (mean + std * eps) * 0.18215 by exactly s.
            scaled = block * np.float32(s)
            out_file = os.path.join(out_dirs[t], cls + ".npy")
            tmp = out_file[:-4] + f".tmp.{os.getpid()}.npy"
            np.save(tmp, scaled)
            os.replace(tmp, out_file)
            written[t] += scaled.shape[0]
        done += len(rows)
        rate = done / (time() - t0 + 1e-6)
        print(f"\r  writing: {done:,}/{total_rows:,} rows ({rate:.0f} rows/s)",
              end="", flush=True)
    print()

    # ── 5. manifest + per-subset metadata ────────────────────────────────────
    manifest_classes = {}
    for cls, rows, names in zip(classes, selection, filenames_per_class):
        if len(rows) == 0:
            continue
        manifest_classes[cls] = {
            "rows": rows.tolist(),
            "files": [names[i] for i in rows] if names is not None else None,
        }
    manifest = {
        "source_packed_latent_path": args.packed_latent_path,
        "source_image_path": args.image_path,
        "source_mode": args.source,
        "sampling": args.sampling,
        "num_images_requested": args.num_images,
        "num_images_selected": n_selected,
        "seed": args.seed,
        "latent_shape": list(latent_shape),
        "classes": manifest_classes,
    }
    with open(os.path.join(args.output_root, "subset_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 70)
    for t, s in scales.items():
        achieved = sigma_measured * s
        info = {
            "dataset_name": (f"{args.name_prefix}-base" if t is None
                             else f"{args.name_prefix}-std{fmt_std(t)}"),
            "target_std": t,
            "source_std": source_std,
            "source_std_origin": source_std_origin,
            "sigma_measured_unscaled": sigma_measured,
            "latent_scale": s,
            "achieved_std_predicted": achieved,
            "num_images": written[t],
            "num_classes": sum(1 for r in selection if len(r) > 0),
            "latent_shape": list(latent_shape),
            "latent_scale_constant": LATENT_SCALE,
            "packed_latent_path": out_dirs[t],
            "note": ("Latents are stored pre-scaled: z = (mean + std*eps) * 0.18215 "
                     "already includes latent_scale. Divide generated latents by "
                     "latent_scale before VAE decoding."),
        }
        if args.verify:
            v_classes, v_counts, _ = scan_packed_classes(out_dirs[t])
            v_selection = [np.arange(n, dtype=np.int64) for n in v_counts]
            v_acc = measure_sigma(out_dirs[t], v_classes, v_selection, latent_shape,
                                  device, args.orientation, args.seed)
            info["achieved_std_measured"] = v_acc.sigma()
        with open(os.path.join(os.path.dirname(os.path.dirname(out_dirs[t])),
                               "scale_info.json"), "w") as f:
            json.dump(info, f, indent=2)

        label = "base" if t is None else f"std {t:g}"
        line = (f" {label:>10}  scale = {s:.6f}  images = {written[t]:,}  "
                f"predicted sigma = {achieved:.6f}")
        if args.verify:
            line += f"  measured sigma = {info['achieved_std_measured']:.6f}"
        print(line)
    print("=" * 70)
    print(f"Manifest    -> {os.path.join(args.output_root, 'subset_manifest.json')}")
    print(f"Base stats  -> {os.path.join(args.output_root, 'base_stats.json')}")
    print("Next: decode_latents.py to produce the FID reference images.")


if __name__ == "__main__":
    main()
