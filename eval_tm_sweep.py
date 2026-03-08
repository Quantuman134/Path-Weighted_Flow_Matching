"""
Two-model TM sweep evaluation for SiT.

For each tm in tm_list:
  - Model 1 generates latents from t=0 → tm  (stage 1)
  - Model 2 refines latents from tm → t=1    (stage 2)
Special cases:
  tm=0.0 → only model2, full ODE t=0→1
  tm=1.0 → only model1, full ODE t=0→1

Output: FID-vs-tm curve + grid images saved under ./experiment/<name>/
"""

import argparse
import json
import math
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from diffusers.models import AutoencoderKL
from scipy.linalg import sqrtm
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Ensure the SiT directory is on the path when run from elsewhere
_SIT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SIT_DIR not in sys.path:
    sys.path.insert(0, _SIT_DIR)

from FID import get_inception_features
from models import SiT_models
from transport import Sampler, create_transport


# ─── Config ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Model loading ───────────────────────────────────────────────────────────

def load_sit_model(model_cfg: dict, device: str):
    """Load a SiT model from a checkpoint. Handles EMA and plain weights."""
    latent_size = model_cfg["image_size"] // 8
    model = SiT_models[model_cfg["model"]](
        input_size=latent_size,
        num_classes=model_cfg["num_classes"],
    ).to(device)

    state = torch.load(model_cfg["ckpt"], map_location="cpu", weights_only=False)
    key = "ema" if model_cfg.get("use_ema", True) else "model"
    model.load_state_dict(state[key])
    model.eval()
    print(f"  Loaded {model_cfg['model']} from {model_cfg['ckpt']} (key='{key}')")
    return model


def build_sampler(model_cfg: dict) -> Sampler:
    transport = create_transport(
        path_type=model_cfg.get("path_type", "Linear"),
        prediction=model_cfg.get("prediction", "velocity"),
        loss_weight=model_cfg.get("loss_weight", None),
        train_eps=model_cfg.get("train_eps", None),
        sample_eps=model_cfg.get("sample_eps", None),
        t_min=0.0,
    )
    return Sampler(transport)


# ─── Reference features ──────────────────────────────────────────────────────

def get_reference_features(
    imagenet_val_path: str,
    num_ref: int,
    image_size: int,
    batch_size: int,
    device: str,
    seed: int = 0,
) -> np.ndarray:
    """Load ImageNet val images and extract Inception features (computed once)."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),          # → [0, 1]
    ])
    dataset = datasets.ImageFolder(imagenet_val_path, transform=transform)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(num_ref, len(dataset)), replace=False)
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    imgs = torch.cat([x for x, _ in loader], dim=0)
    print(f"  Extracting Inception features for {len(imgs)} reference images...")
    return get_inception_features(imgs, batch_size=batch_size, device=device)


# ─── Class label generation ──────────────────────────────────────────────────

def make_balanced_labels(num_images: int, num_classes: int, seed: int = 0) -> np.ndarray:
    """Return a (num_images,) array of class labels balanced across all classes."""
    base = num_images // num_classes
    remainder = num_images % num_classes
    labels = []
    for c in range(num_classes):
        labels.extend([c] * (base + (1 if c < remainder else 0)))
    labels = np.array(labels, dtype=np.int64)
    np.random.default_rng(seed).shuffle(labels)
    return labels


# ─── Generation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_batch(
    z: torch.Tensor,
    y: torch.Tensor,
    tm: float,
    model1, model2,
    sampler1: Sampler, sampler2: Sampler,
    sampling_cfg: dict,
    vae,
    device: str,
) -> torch.Tensor:
    """
    Generate one batch using the two-stage handoff at tm.

    Args:
        z: (B, 4, H, W) initial Gaussian noise
        y: (B,) class labels
    Returns:
        (B, 3, H, W) decoded images in [0, 1] on CPU
    """
    n = z.shape[0]
    num_steps = sampling_cfg["num_sampling_steps"]
    method    = sampling_cfg["sampling_method"]
    atol      = sampling_cfg["atol"]
    rtol      = sampling_cfg["rtol"]
    cfg_scale = sampling_cfg["cfg_scale"]
    use_cfg   = cfg_scale > 1.0

    if use_cfg:
        z      = torch.cat([z, z], dim=0)
        y_null = torch.tensor([1000] * n, device=device)
        y      = torch.cat([y, y_null], dim=0)
        model_kwargs   = dict(y=y, cfg_scale=cfg_scale)
        model1_forward = model1.forward_with_cfg
        model2_forward = model2.forward_with_cfg
    else:
        model_kwargs   = dict(y=y)
        model1_forward = model1.forward
        model2_forward = model2.forward

    if tm == 0.0:
        # Only model2: full ODE from t=0 to t=1
        fn = sampler2.sample_ode(
            sampling_method=method, num_steps=num_steps, atol=atol, rtol=rtol
        )
        samples = fn(z, model2_forward, **model_kwargs)[-1]

    elif tm == 1.0:
        # Only model1: full ODE from t=0 to t=1
        fn = sampler1.sample_ode(
            sampling_method=method, num_steps=num_steps, atol=atol, rtol=rtol
        )
        samples = fn(z, model1_forward, **model_kwargs)[-1]

    else:
        # Stage 1: t=0 → tm  (model1)
        steps1 = max(1, round(num_steps * tm))
        steps2 = max(1, num_steps - steps1)

        fn1 = sampler1.sample_ode(
            sampling_method=method, num_steps=steps1, atol=atol, rtol=rtol,
            t1_override=tm,
        )
        x_m = fn1(z, model1_forward, **model_kwargs)[-1]

        # Stage 2: tm → t=1  (model2)
        fn2 = sampler2.sample_ode(
            sampling_method=method, num_steps=steps2, atol=atol, rtol=rtol,
            t0_override=tm,
        )
        samples = fn2(x_m, model2_forward, **model_kwargs)[-1]

    if use_cfg:
        samples, _ = samples.chunk(2, dim=0)

    # Decode latents → pixel space
    decoded = vae.decode(samples / 0.18215).sample   # (-1, 1)
    decoded = (decoded.clamp(-1.0, 1.0) + 1.0) / 2.0 # [0, 1]
    return decoded.cpu()


@torch.no_grad()
def generate_for_tm(
    tm: float,
    model1, model2,
    sampler1: Sampler, sampler2: Sampler,
    vae,
    all_labels: np.ndarray,
    sampling_cfg: dict,
    batch_size: int,
    latent_size: int,
    device: str,
) -> torch.Tensor:
    """Generate all images for a given tm. Returns (N, 3, H, W) in [0, 1]."""
    n = len(all_labels)
    all_images = []

    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        y     = torch.tensor(all_labels[start:end], device=device)
        z     = torch.randn(end - start, 4, latent_size, latent_size, device=device)
        imgs  = generate_batch(
            z, y, tm, model1, model2, sampler1, sampler2,
            sampling_cfg, vae, device
        )
        all_images.append(imgs)
        print(f"\r  Generated {end}/{n} images", end="", flush=True)

    print()
    return torch.cat(all_images, dim=0)


# ─── FID ─────────────────────────────────────────────────────────────────────

def compute_fid(gen_features: np.ndarray, ref_features: np.ndarray) -> float:
    """Fréchet Inception Distance from pre-extracted feature arrays."""
    mu_r, mu_g     = ref_features.mean(axis=0), gen_features.mean(axis=0)
    sigma_r         = np.cov(ref_features, rowvar=False)
    sigma_g         = np.cov(gen_features, rowvar=False)
    covmean, _      = sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_r - mu_g
    return float(diff @ diff + np.trace(sigma_r + sigma_g - 2.0 * covmean))


# ─── Visualisation ───────────────────────────────────────────────────────────

def save_grid(images: torch.Tensor, grid_samples: int, path: str):
    grid_imgs = images[:grid_samples]
    nrow = int(math.sqrt(grid_samples))
    save_image(grid_imgs, path, nrow=nrow)
    print(f"  Saved grid  → {path}")


def plot_fid_curve(fid_results: dict, save_path: str):
    tms  = sorted(fid_results.keys())
    fids = [fid_results[t] for t in tms]
    plt.figure(figsize=(8, 5))
    plt.plot(tms, fids, marker="o", linewidth=2)
    plt.xlabel("tm  (handoff timestep)")
    plt.ylabel("FID")
    plt.title("FID vs Handoff Timestep tm")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved FID curve → {save_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TM-sweep evaluation for two-stage SiT")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device from config (e.g. cuda:1)")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = args.device or cfg.get("device", "cuda")

    # ── Experiment directory ──────────────────────────────────────────────────
    exp_dir    = os.path.join(cfg["experiment"]["results_base_dir"],
                              cfg["experiment"]["name"])
    images_dir = os.path.join(exp_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Save a copy of the config for reproducibility
    shutil.copy(args.config, os.path.join(exp_dir, "config.yaml"))
    print(f"Experiment dir: {exp_dir}")

    seed = cfg["eval"].get("seed", 0)
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)

    # ── Load models ───────────────────────────────────────────────────────────
    print("\nLoading Model 1 (stage 1: t=0 → tm)...")
    model1   = load_sit_model(cfg["model1"], device)
    sampler1 = build_sampler(cfg["model1"])

    print("Loading Model 2 (stage 2: tm → t=1)...")
    model2   = load_sit_model(cfg["model2"], device)
    sampler2 = build_sampler(cfg["model2"])

    # ── Load VAE ──────────────────────────────────────────────────────────────
    vae_name = cfg.get("vae", "ema")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_name}").to(device)
    print(f"Loaded VAE (sd-vae-ft-{vae_name})")

    # ── Reference features (computed once) ────────────────────────────────────
    batch_size = cfg["eval"]["batch_size"]
    image_size = cfg["model1"]["image_size"]
    num_ref    = cfg["data"]["num_ref_images"]
    print(f"\nBuilding reference features from {cfg['data']['imagenet_val_path']} ...")
    ref_features = get_reference_features(
        cfg["data"]["imagenet_val_path"],
        num_ref, image_size, batch_size, device, seed=seed,
    )

    # ── Class labels (balanced across 1000 classes) ───────────────────────────
    num_images  = cfg["eval"]["num_images"]
    num_classes = cfg["model1"]["num_classes"]
    all_labels  = make_balanced_labels(num_images, num_classes, seed=seed)

    latent_size = image_size // 8

    # ── TM sweep ──────────────────────────────────────────────────────────────
    fid_results: dict[float, float] = {}
    tm_list = cfg["eval"]["tm_list"]

    for tm in tm_list:
        print(f"\n{'─'*50}")
        print(f" tm = {tm:.2f}")
        print(f"{'─'*50}")

        gen_images = generate_for_tm(
            tm, model1, model2, sampler1, sampler2, vae,
            all_labels, cfg["sampling"], batch_size, latent_size, device,
        )

        print("  Extracting Inception features for FID...")
        gen_features = get_inception_features(gen_images, batch_size=batch_size, device=device)
        fid = compute_fid(gen_features, ref_features)
        fid_results[tm] = fid
        print(f"  FID = {fid:.4f}")

        tm_str    = f"{tm:.2f}".replace(".", "_")
        grid_path = os.path.join(images_dir, f"grid_tm_{tm_str}.png")
        save_grid(gen_images, cfg["eval"]["grid_samples"], grid_path)

    # ── Save results ──────────────────────────────────────────────────────────
    json_path = os.path.join(exp_dir, "fid_results.json")
    with open(json_path, "w") as f:
        json.dump({str(k): v for k, v in sorted(fid_results.items())}, f, indent=2)
    print(f"\nSaved FID results → {json_path}")

    plot_fid_curve(fid_results, os.path.join(exp_dir, "fid_curve.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
