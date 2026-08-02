"""
Estimate the variance scale of VAE latents on CIFAR-10 or ImageNet (single GPU).

Procedure
---------
1. Randomly sample N latents (default 10000) from the dataset.
2. Obtain the latent posterior q(z|x) with the same pipeline used by SiT
   training.  Two sources are supported:
     - 'images': raw ImageFolder -> center_crop_arr -> [-1,1] -> AutoencoderKL
     - 'packed': the pre-encoded per-class .npy latents (mean/std) produced by
       encode_dataset.py + pack_latents.py -- identical preprocessing, no VAE
       forward pass needed.
   Either way the latent is scaled by 0.18215, exactly as in train.py.
3. Compute  sigma = sqrt( (1/d) * tr(Cov(z)) ),  where z is the flattened latent
   code and d = C * H * W is its dimension.  Note that

       (1/d) tr(Cov(z)) = (1/d) sum_i Var(z_i) = mean over dimensions of the
       per-dimension variance,

   so sigma is the RMS per-coordinate standard deviation of the latent code.
4. Compute  (1/d) * E_{p_t}[ ||z_t||^2 ]  along the training interpolant, where
   z_t = alpha_t * z + sigma_t * eps  and eps ~ N(0, I) is independent of z
   (transport/path.py). Because the cross term vanishes this equals

       (1/d) E||z_t||^2 = alpha_t^2 * (1/d)E||z||^2 + sigma_t^2,

   which is reported on a t-grid together with an independent Monte-Carlo
   estimate as a cross-check.
5. Save the result to a JSON summary (+ an .npz with the per-dimension moments).

The statistics are accumulated in a streaming fashion (float64 running sums of
z and z^2), so memory stays O(d) regardless of N.

Usage -- CIFAR-10 (raw images + VAE):
  python exp_latent_variance.py \
      --data_path /scratch/project/prj-02-visual-ai/hkzhang/cifar10/cifar10_imagefolder/train \
      --num_images 10000 --image_size 256 --vae ema --device cuda:0

Usage -- ImageNet (pre-encoded packed latents, no VAE needed):
  python exp_latent_variance.py \
      --packed_latent_path /scratch/project/prj-02-visual-ai/hkzhang/ILSVRC/latents_packed/train \
      --num_images 10000 --tag imagenet --device cuda:0
"""
import os
import json
import bisect
import argparse
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image


# SD-VAE latent scaling factor used everywhere in this repo (train.py, sample.py).
LATENT_SCALE = 0.18215


# ── preprocessing (identical to train.py / encode_dataset.py) ────────────────

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


# ── packed-latent source (mirrors train.py's PackedLatentImageFolder) ────────

class PackedLatentSource:
    """Random access into the per-class packed latent .npy files.

    Each <class_name>.npy has shape (N, 2, 2, 4, H, W) float32:
      axis 1: orientation -- 0 = original, 1 = horizontally flipped
      axis 2: parameter   -- 0 = mean,     1 = std

    Only the posterior parameters are stored, so the latent is reconstructed as
    z = (mean + std * eps) * 0.18215 -- the same expression train.py uses.
    """

    def __init__(self, root):
        npy_files = sorted(f for f in os.listdir(root) if f.endswith(".npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy latent files found in {root}")
        self.classes = [os.path.splitext(f)[0] for f in npy_files]

        self._paths = []
        self._cum_sizes = []
        cumsum = 0
        latent_shape = None
        for fname in npy_files:
            path = os.path.join(root, fname)
            arr = np.load(path, mmap_mode="r")
            if latent_shape is None:
                latent_shape = tuple(arr.shape[3:])   # (4, H, W)
            elif tuple(arr.shape[3:]) != latent_shape:
                raise ValueError(
                    f"{fname} has latent shape {tuple(arr.shape[3:])}, "
                    f"expected {latent_shape}"
                )
            cumsum += arr.shape[0]
            del arr                                   # release the mmap
            self._paths.append(path)
            self._cum_sizes.append(cumsum)
        self._total = cumsum
        self.latent_shape = latent_shape
        self._mmaps = {}

    def __len__(self):
        return self._total

    def _locate(self, index):
        """Flat index -> (file position, index within that file)."""
        pos = bisect.bisect_right(self._cum_sizes, index)
        local = index - (self._cum_sizes[pos - 1] if pos > 0 else 0)
        return pos, local

    def get_batch(self, indices, orientation=0):
        """Returns (mean, std) float32 tensors of shape (len(indices), 4, H, W)."""
        means, stds = [], []
        for idx in indices:
            pos, local = self._locate(int(idx))
            path = self._paths[pos]
            if path not in self._mmaps:
                self._mmaps[path] = np.load(path, mmap_mode="r")
            arr = self._mmaps[path]
            means.append(torch.from_numpy(arr[local, orientation, 0].copy()))
            stds.append(torch.from_numpy(arr[local, orientation, 1].copy()))
        return torch.stack(means), torch.stack(stds)


# ── posterior (mean, std) iterators -- one per data source ───────────────────

def iter_posterior_from_images(loader, vae, device):
    """Yields un-scaled (mean, std) batches by running the VAE encoder."""
    with torch.no_grad():
        for x, _y in loader:
            x = x.to(device, non_blocking=True)
            posterior = vae.encode(x).latent_dist
            yield posterior.mean, posterior.std


def iter_posterior_from_packed(source, indices, batch_size, device, orientation=0):
    """Yields un-scaled (mean, std) batches read from the packed .npy latents."""
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        mean, std = source.get_batch(chunk, orientation=orientation)
        yield mean.to(device, non_blocking=True), std.to(device, non_blocking=True)


# ── streaming moment accumulator ─────────────────────────────────────────────

class MomentAccumulator:
    """Accumulates per-dimension sum(z) and sum(z^2) in float64."""

    def __init__(self, dim, device):
        self.dim = dim
        self.n = 0
        self.sum = torch.zeros(dim, dtype=torch.float64, device=device)
        self.sum_sq = torch.zeros(dim, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, z):
        """z: (B, C, H, W) tensor of latents."""
        zf = z.reshape(z.shape[0], -1).to(torch.float64)
        self.n += zf.shape[0]
        self.sum += zf.sum(dim=0)
        self.sum_sq += (zf * zf).sum(dim=0)

    def finalize(self):
        """Returns (per_dim_mean, per_dim_var) as float64 CPU tensors.

        Uses the unbiased (n-1) estimator for the variance.
        """
        assert self.n > 1, "Need at least 2 samples to estimate a variance."
        mean = self.sum / self.n
        # sum((z - mu)^2) = sum(z^2) - n * mu^2  -> clamp guards fp round-off.
        sq_dev = (self.sum_sq - self.n * mean * mean).clamp_min(0.0)
        var = sq_dev / (self.n - 1)
        return mean.cpu(), var.cpu()

    def second_moment_per_dim(self):
        """(1/d) E[||z||^2], computed directly from the raw sum of squares."""
        assert self.n > 0
        return float(self.sum_sq.sum().item() / (self.n * self.dim))


class PathNormAccumulator:
    """Monte-Carlo estimate of (1/d) E_{p_t}[||z_t||^2] on a grid of t.

    z_t follows the interpolant used for training:  z_t = alpha_t * z + sigma_t * eps,
    with eps ~ N(0, I) drawn independently of z (see transport/path.py).
    """

    def __init__(self, alphas, sigmas, dim, device):
        self.alphas = alphas.to(device=device, dtype=torch.float64)   # (T,)
        self.sigmas = sigmas.to(device=device, dtype=torch.float64)
        self.dim = dim
        self.n = 0
        self.sum = torch.zeros(len(alphas), dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, z, generator):
        """z: (B, C, H, W) latent batch (already scaled by 0.18215)."""
        zf = z.reshape(z.shape[0], -1).to(torch.float64)
        self.n += zf.shape[0]
        for k in range(len(self.alphas)):
            eps = torch.randn(zf.shape, generator=generator,
                              device=zf.device, dtype=zf.dtype)
            zt = self.alphas[k] * zf + self.sigmas[k] * eps
            self.sum[k] += (zt * zt).sum() / self.dim

    def finalize(self):
        """Returns the per-t estimate of (1/d) E[||z_t||^2] as a float64 CPU tensor."""
        assert self.n > 0
        return (self.sum / self.n).cpu()


def summarize(mean, var, shape, scale, n):
    """Build the statistics dict for one accumulator.

    `m` denotes the per-dimension mean vector E[z] (length d); `n` is the sample
    count, needed to turn the unbiased per-dim variance back into the biased one
    so the reported moments are the raw ones.

    Reports exactly three quantities:
      1. (1/d) E||z||^2
      2. (1/d) sum_i m_i          -- the scalar average latent value
      3. (1/d) E||z||^2 - (1/d) ||m||^2
    """
    d = mean.numel()
    var_biased = var * (n - 1) / n                       # E[z_i^2] - m_i^2

    e_z_sq_per_dim = float((var_biased + mean * mean).sum().item()) / d
    m_per_dim = float(mean.mean().item())
    m_norm_sq_per_dim = float((mean * mean).sum().item()) / d

    return {
        "d": int(d),
        "E_z_sq_per_dim": e_z_sq_per_dim,                            # 1
        "m_per_dim": m_per_dim,                                      # 2
        "E_z_sq_minus_m_norm_sq_per_dim": e_z_sq_per_dim - m_norm_sq_per_dim,  # 3
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Estimate sqrt((1/d) tr(Cov(z))) of VAE latents (CIFAR-10 / ImageNet)"
    )
    parser.add_argument("--data_path", type=str,
                        default="/scratch/project/prj-02-visual-ai/hkzhang/cifar10/cifar10_imagefolder/train",
                        help="Root of the ImageFolder dataset (used when "
                             "--packed_latent_path is not given)")
    parser.add_argument("--packed_latent_path", type=str, default=None,
                        help="Root of pre-encoded packed latents (<class>.npy). "
                             "If given, latents are read from disk and no VAE is run.")
    parser.add_argument("--orientation", type=int, default=0, choices=[0, 1],
                        help="Packed latents only: 0 = original image, "
                             "1 = horizontally flipped. Default 0 (no flip augmentation).")
    parser.add_argument("--num_images", type=int, default=10000,
                        help="Number of images to randomly sample")
    parser.add_argument("--path_type", type=str, default="Linear",
                        choices=["Linear", "GVP", "VP"],
                        help="Interpolant used for the E||z_t||^2 curve "
                             "(alpha_t/sigma_t come from transport/path.py)")
    parser.add_argument("--num_t", type=int, default=21,
                        help="Number of t grid points in [0,1] for E||z_t||^2")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size fed to the VAE (must match training)")
    parser.add_argument("--vae", type=str, default="ema", choices=["ema", "mse"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the random image subset and the VAE sampling")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="./results/latent_variance")
    parser.add_argument("--tag", type=str, default="cifar10",
                        help="Short tag used in the output file names")
    parser.add_argument("--save_per_dim", action="store_true",
                        help="Also dump the per-dimension mean/variance arrays to .npz")
    args = parser.parse_args()

    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (VAE downsamples 8x)."

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── data source: packed latents (preferred, no VAE) or raw images ────────
    use_packed = args.packed_latent_path is not None
    if use_packed and not os.path.isdir(args.packed_latent_path):
        raise FileNotFoundError(
            f"packed_latent_path does not exist: {args.packed_latent_path}"
        )
    rng = np.random.default_rng(args.seed)

    if use_packed:
        source = PackedLatentSource(args.packed_latent_path)
        total = len(source)
        n_take = min(args.num_images, total)
        # Sorted indices keep the mmap reads roughly sequential per class file;
        # the *sample* itself is still a uniform draw without replacement.
        indices = np.sort(rng.permutation(total)[:n_take])
        latent_shape = source.latent_shape
        data_source = args.packed_latent_path

        print(f"Dataset : {data_source}  (packed latents)")
        print(f"          {total:,} latents in {len(source.classes)} classes")
        print(f"Sampled : {n_take:,} latents (seed={args.seed}, "
              f"orientation={'flipped' if args.orientation else 'original'})")
        vae = None
        batches = iter_posterior_from_packed(
            source, indices, args.batch_size, device, orientation=args.orientation
        )
    else:
        transform = transforms.Compose([
            CenterCropTransform(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        dataset = ImageFolder(args.data_path, transform=transform)
        total = len(dataset)
        n_take = min(args.num_images, total)
        indices = rng.permutation(total)[:n_take]      # uniform sample w/o replacement
        subset = Subset(dataset, indices.tolist())
        latent_shape = (4, args.image_size // 8, args.image_size // 8)
        data_source = args.data_path

        print(f"Dataset : {data_source}")
        print(f"          {total:,} images in {len(dataset.classes)} classes")
        print(f"Sampled : {n_take:,} images (seed={args.seed})")

        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        # Imported lazily so the packed-latent path does not need diffusers.
        from diffusers.models import AutoencoderKL
        print(f"Loading VAE (sd-vae-ft-{args.vae}) ...", flush=True)
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
        vae.eval()
        batches = iter_posterior_from_images(loader, vae, device)

    if n_take < args.num_images:
        print(f"[warn] requested {args.num_images} samples but the dataset has only "
              f"{total}; using {n_take}.")

    d = int(np.prod(latent_shape))
    print(f"Latent  : shape {latent_shape}, d = {d}")

    # z ~ q(z|x), exactly what train.py feeds the transformer.
    acc_sample = MomentAccumulator(d, device)

    # Interpolant coefficients, taken straight from the training code so the
    # curve matches whatever path SiT is trained with.
    from transport.path import ICPlan, GVPCPlan, VPCPlan
    plan = {"Linear": ICPlan, "GVP": GVPCPlan, "VP": VPCPlan}[args.path_type]()
    t_grid = torch.linspace(0.0, 1.0, args.num_t, dtype=torch.float64)
    alphas, _ = plan.compute_alpha_t(t_grid)
    sigmas, _ = plan.compute_sigma_t(t_grid)
    alphas = torch.as_tensor(alphas, dtype=torch.float64)
    sigmas = torch.as_tensor(sigmas, dtype=torch.float64)
    acc_path = PathNormAccumulator(alphas, sigmas, d, device)
    print(f"Path    : {args.path_type}, {args.num_t} t-grid points in [0, 1]")

    # Deterministic posterior sampling noise, independent of worker/read order.
    generator = torch.Generator(device=device).manual_seed(args.seed)
    # Separate stream for the interpolant noise so adding the E||z_t||^2 curve
    # does not perturb the latent samples the variance estimate is built from.
    path_generator = torch.Generator(device=device).manual_seed(args.seed + 1)

    t0 = time()
    done = 0
    with torch.no_grad():
        for mu, std in batches:
            noise = torch.randn(mu.shape, generator=generator,
                                device=device, dtype=mu.dtype)
            z_sample = (mu + std * noise) * LATENT_SCALE

            acc_sample.update(z_sample)
            acc_path.update(z_sample, path_generator)

            done += mu.shape[0]
            elapsed = time() - t0 + 1e-6
            print(f"\r  {done:,}/{n_take:,}  ({done / elapsed:.1f} img/s)",
                  end="", flush=True)
    print()
    assert done == n_take, f"processed {done} samples but expected {n_take}"

    # ── statistics ───────────────────────────────────────────────────────────
    mean_s, var_s = acc_sample.finalize()
    stats_sample = summarize(mean_s, var_s, latent_shape, LATENT_SCALE, acc_sample.n)

    # ── (1/d) E_{p_t}[||z_t||^2] along the interpolant ───────────────────────
    # z_t = alpha_t * z + sigma_t * eps with eps ~ N(0,I) independent of z, so
    #   (1/d) E||z_t||^2 = alpha_t^2 * (1/d)E||z||^2 + sigma_t^2,
    # the cross term vanishing because E[eps] = 0. The Monte-Carlo estimate is
    # accumulated independently and must agree with this closed form.
    m2 = acc_sample.second_moment_per_dim()          # (1/d) E||z||^2, exact
    analytic = (alphas ** 2) * m2 + sigmas ** 2
    empirical = acc_path.finalize()
    max_dev = float((analytic - empirical).abs().max().item())

    # E over t ~ U(0,1). Integrated on a dense grid, not the (coarse) display
    # grid -- trapezoid on 21 points carries ~1e-3 discretisation error here.
    t_dense = torch.linspace(0.0, 1.0, 2001, dtype=torch.float64)
    a_dense, _ = plan.compute_alpha_t(t_dense)
    s_dense, _ = plan.compute_sigma_t(t_dense)
    a_dense = torch.as_tensor(a_dense, dtype=torch.float64)
    s_dense = torch.as_tensor(s_dense, dtype=torch.float64)
    t_avg = float(torch.trapz(a_dense ** 2 * m2 + s_dense ** 2, t_dense).item())

    path_stats = {
        "path_type": args.path_type,
        "definition": "z_t = alpha_t * z + sigma_t * eps,  eps ~ N(0, I)",
        "E_z_sq_per_dim": m2,                        # (1/d) E||z||^2  (t = 1)
        "t_grid": [float(v) for v in t_grid],
        "alpha_t": [float(v) for v in alphas],
        "sigma_t": [float(v) for v in sigmas],
        "E_zt_sq_per_dim_analytic": [float(v) for v in analytic],
        "E_zt_sq_per_dim_empirical": [float(v) for v in empirical],
        "max_abs_deviation": max_dev,
        "t_averaged_uniform": t_avg,                 # int_0^1 (1/d)E||z_t||^2 dt
    }

    # In packed mode the image size is implied by the stored latent resolution.
    effective_image_size = latent_shape[1] * 8

    result = {
        "experiment": "latent_variance",
        "config": {
            "source": "packed_latents" if use_packed else "images+vae",
            "data_path": data_source,
            "dataset_size": total,
            "num_images": n_take,
            "image_size": effective_image_size,
            # For packed latents this is the VAE that encode_dataset.py used.
            "vae": f"sd-vae-ft-{args.vae}",
            "orientation": args.orientation if use_packed else 0,
            "latent_scale": LATENT_SCALE,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": str(device),
        },
        # z ~ q(z|x): the latents train.py actually feeds the transformer.
        "latents_sampled": stats_sample,
        "path_second_moment": path_stats,
    }

    out_json = os.path.join(args.output_dir, f"latent_variance_{args.tag}_{args.vae}.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    if args.save_per_dim:
        out_npz = os.path.join(args.output_dir, f"latent_variance_{args.tag}_{args.vae}_perdim.npz")
        np.savez(
            out_npz,
            mean_sampled=mean_s.numpy().reshape(latent_shape),
            var_sampled=var_s.numpy().reshape(latent_shape),
            indices=indices,
            t_grid=t_grid.numpy(),
            E_zt_sq_analytic=analytic.numpy(),
            E_zt_sq_empirical=empirical.numpy(),
        )
        print(f"Per-dim moments -> {out_npz}")

    # ── report ───────────────────────────────────────────────────────────────
    print("=" * 66)
    print(f" images = {n_take:,}   d = {d}   VAE = sd-vae-ft-{args.vae}   "
          f"image_size = {effective_image_size}   "
          f"source = {'packed latents' if use_packed else 'images+VAE'}")
    print("-" * 66)
    s = stats_sample
    print(f"   (1/d) E||z||^2                 = {s['E_z_sq_per_dim']:.6f}")
    print(f"   (1/d) m                        = {s['m_per_dim']:.6f}")
    print(f"   (1/d) E||z||^2 - (1/d)||m||^2  = "
          f"{s['E_z_sq_minus_m_norm_sq_per_dim']:.6f}")
    print("-" * 66)
    print(f" (1/d) E_p_t[||z_t||^2]   path = {args.path_type}   "
          f"z_t = alpha_t z + sigma_t eps")
    print(f"   (1/d) E||z||^2 (t=1)          = {m2:.6f}")
    print(f"     t     alpha_t  sigma_t    analytic   empirical")
    for k in range(len(t_grid)):
        print(f"   {float(t_grid[k]):.3f}    {float(alphas[k]):.4f}   "
              f"{float(sigmas[k]):.4f}    {float(analytic[k]):9.6f}   "
              f"{float(empirical[k]):9.6f}")
    print(f"   max |analytic - empirical|    = {max_dev:.2e}")
    print(f"   mean over t ~ U(0,1)          = {t_avg:.6f}")
    print("=" * 66)
    print(f"Saved -> {out_json}")


if __name__ == "__main__":
    main()
