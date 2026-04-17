"""
Alpha-sweep evaluation for velocity-blended SiT.

For each alpha in alpha_list:
  At every ODE step, both models are queried and their velocity predictions blended:
    v_t = alpha * v1_t + (1 - alpha) * v2_t
  where model1 and model2 can use any supported prediction type (target, velocity,
  noise, score) — each model's drift function handles the conversion to velocity
  internally.

Supported blend combinations (set via config prediction fields):
  target   + velocity  (default)
  target   + noise
  velocity + noise
  velocity + velocity
  (and any other combination of the four prediction types)

Output: FID-vs-alpha curve + grid images saved under ./experiment/<name>/
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
import torch.distributed as dist
import yaml
from diffusers.models import AutoencoderKL
from scipy.linalg import sqrtm
from torchvision import datasets, transforms
from torchvision.utils import save_image

_SIT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SIT_DIR not in sys.path:
    sys.path.insert(0, _SIT_DIR)

from FID import get_inception_features
from IS import get_inception_probs, compute_is_from_probs
from models import SiT_models
from transport import Sampler, create_transport


# ─── Blended velocity model ──────────────────────────────────────────────────

class BlendedVelocityModel:
    """
    Wraps two SiT models and blends their velocity predictions at every step:
        v_t = alpha * v1_t + (1 - alpha) * v2_t

    The per-model drift functions (from transport.get_drift()) handle each
    model's prediction type internally (e.g. target_ode for model1,
    velocity_ode for model2), so no manual conversion is needed here.
    """

    def __init__(self, model1, model2, alpha, drift1, drift2):
        self.model1 = model1
        self.model2 = model2
        self.alpha  = alpha
        self.drift1 = drift1   # transport1.get_drift() — handles target→velocity internally
        self.drift2 = drift2   # transport2.get_drift() — returns velocity directly

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        a = self.alpha(t.flatten()[0].item()) if callable(self.alpha) else self.alpha
        v1 = self.drift1(x, t, self.model1, **kwargs)
        v2 = self.drift2(x, t, self.model2, **kwargs)
        return a * v1 + (1.0 - a) * v2

    def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        # x is a doubled batch (2N); forward_with_cfg returns CFG-combined output of same shape
        a = self.alpha(t.flatten()[0].item()) if callable(self.alpha) else self.alpha
        v1 = self.drift1(x, t, self.model1.forward_with_cfg, **kwargs)
        v2 = self.drift2(x, t, self.model2.forward_with_cfg, **kwargs)
        return a * v1 + (1.0 - a) * v2


# ─── Alpha scheduler ─────────────────────────────────────────────────────────

def build_alpha_fn(scheduler_type):
    """Return a callable alpha(t: float) -> float, or None for constant mode."""
    if scheduler_type is None:
        return None
    if scheduler_type == "linear":
        return lambda t: float(1-t)
    raise ValueError(f"Unknown alpha_scheduler: {scheduler_type!r}. Use 'linear' or null.")


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


def build_blend_sampler(model_cfg: dict) -> Sampler:
    """Build a velocity-type Sampler for full t ∈ [0, 1] ODE integration.
    Uses the path_type from model_cfg so it matches the models' interpolant."""
    transport = create_transport(
        path_type=model_cfg.get("path_type", "Linear"),
        prediction="velocity",
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
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
        transforms.ToTensor(),
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
def generate_batch_blended(
    z: torch.Tensor,
    y: torch.Tensor,
    alpha,
    model1, model2,
    sampler1: Sampler, sampler2: Sampler,
    blend_sampler: Sampler,
    sampling_cfg: dict,
    vae,
    device: str,
) -> torch.Tensor:
    """
    Generate one batch using velocity blending with weight alpha.

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

    blended = BlendedVelocityModel(model1, model2, alpha, sampler1.drift, sampler2.drift)

    if use_cfg:
        z      = torch.cat([z, z], dim=0)
        y_null = torch.tensor([1000] * n, device=device)
        y      = torch.cat([y, y_null], dim=0)
        model_kwargs  = dict(y=y, cfg_scale=cfg_scale)
        model_forward = blended.forward_with_cfg
    else:
        model_kwargs  = dict(y=y)
        model_forward = blended.forward

    fn = blend_sampler.sample_ode(
        sampling_method=method, num_steps=num_steps, atol=atol, rtol=rtol
    )
    samples = fn(z, model_forward, **model_kwargs)[-1]

    if use_cfg:
        samples, _ = samples.chunk(2, dim=0)

    decoded = vae.decode(samples / 0.18215).sample
    decoded = (decoded.clamp(-1.0, 1.0) + 1.0) / 2.0
    return decoded.cpu()


@torch.no_grad()
def generate_for_alpha(
    alpha,
    model1, model2,
    sampler1: Sampler, sampler2: Sampler,
    blend_sampler: Sampler,
    vae,
    all_labels: np.ndarray,
    sampling_cfg: dict,
    batch_size: int,
    latent_size: int,
    device: str,
) -> torch.Tensor:
    """Generate all images for a given alpha. Returns (N, 3, H, W) in [0, 1]."""
    n = len(all_labels)
    all_images = []

    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        y    = torch.tensor(all_labels[start:end], device=device)
        z    = torch.randn(end - start, 4, latent_size, latent_size, device=device)
        imgs = generate_batch_blended(
            z, y, alpha, model1, model2, sampler1, sampler2, blend_sampler,
            sampling_cfg, vae, device,
        )
        all_images.append(imgs)
        print(f"\r  Generated {end}/{n} images", end="", flush=True)

    print()
    return torch.cat(all_images, dim=0)


# ─── FID ─────────────────────────────────────────────────────────────────────

def compute_fid(gen_features: np.ndarray, ref_features: np.ndarray) -> float:
    """Fréchet Inception Distance from pre-extracted feature arrays."""
    mu_r, mu_g = ref_features.mean(axis=0), gen_features.mean(axis=0)
    sigma_r     = np.cov(ref_features, rowvar=False)
    sigma_g     = np.cov(gen_features, rowvar=False)
    covmean, _  = sqrtm(sigma_r @ sigma_g, disp=False)
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
    alphas = sorted(fid_results.keys())
    fids   = [fid_results[a] for a in alphas]
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, fids, marker="o", linewidth=2)
    plt.xlabel("α  (blend weight for model1 target→velocity)")
    plt.ylabel("FID")
    plt.title("FID vs Blend Weight α\n(α=0: pure velocity model,  α=1: pure target model)")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved FID curve → {save_path}")


def plot_is_curve(is_results: dict, save_path: str):
    alphas = sorted(is_results.keys())
    means  = [is_results[a]["mean"] for a in alphas]
    stds   = [is_results[a]["std"]  for a in alphas]
    plt.figure(figsize=(8, 5))
    plt.errorbar(alphas, means, yerr=stds, marker="o", linewidth=2, capsize=4)
    plt.xlabel("α  (blend weight for model1 target→velocity)")
    plt.ylabel("IS (Inception Score)")
    plt.title("IS vs Blend Weight α\n(α=0: pure velocity model,  α=1: pure target model)")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved IS curve  → {save_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Alpha-sweep evaluation for velocity-blended SiT"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device from config (e.g. cuda:1)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── Distributed init (falls back to single-GPU when not launched via torchrun) ──
    try:
        rank       = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        device  = f"cuda:{local_rank}"
        is_dist = True
    except KeyError:
        rank, world_size, local_rank = 0, 1, 0
        device  = args.device or cfg.get("device", "cuda")
        is_dist = False

    # ── Experiment directory (rank 0 creates; others wait) ───────────────────
    exp_dir    = os.path.join(cfg["experiment"]["results_base_dir"],
                              cfg["experiment"]["name"])
    images_dir = os.path.join(exp_dir, "images")
    if rank == 0:
        os.makedirs(images_dir, exist_ok=True)
        shutil.copy(args.config, os.path.join(exp_dir, "config.yaml"))
        print(f"Experiment dir: {exp_dir}")
    if is_dist:
        dist.barrier()

    seed = cfg["eval"].get("seed", 0)
    torch.manual_seed(seed * world_size + rank)   # unique noise per rank
    torch.set_grad_enabled(False)

    # ── Load models ───────────────────────────────────────────────────────────
    pred1 = cfg["model1"].get("prediction", "velocity")
    pred2 = cfg["model2"].get("prediction", "velocity")

    print(f"\nLoading Model 1 ({pred1} predictor)...")
    model1   = load_sit_model(cfg["model1"], device)
    sampler1 = build_sampler(cfg["model1"])

    print(f"Loading Model 2 ({pred2} predictor)...")
    model2   = load_sit_model(cfg["model2"], device)
    sampler2 = build_sampler(cfg["model2"])

    blend_sampler  = build_blend_sampler(cfg["model1"])

    # ── Load VAE ──────────────────────────────────────────────────────────────
    vae_name = cfg.get("vae", "ema")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_name}").to(device)
    print(f"Loaded VAE (sd-vae-ft-{vae_name})")

    # ── Reference features (rank 0 only — not needed on workers) ────────────
    batch_size = cfg["eval"]["batch_size"]
    image_size = cfg["model1"]["image_size"]
    num_ref    = cfg["data"]["num_ref_images"]
    if rank == 0:
        print(f"\nBuilding reference features from {cfg['data']['imagenet_val_path']} ...")
        ref_features = get_reference_features(
            cfg["data"]["imagenet_val_path"],
            num_ref, image_size, batch_size, device, seed=seed,
        )
    else:
        ref_features = None

    # ── Class labels (balanced across all classes) ────────────────────────────
    num_images  = cfg["eval"]["num_images"]
    num_classes = cfg["model1"]["num_classes"]
    all_labels  = make_balanced_labels(num_images, num_classes, seed=seed)
    rank_labels = all_labels[rank::world_size]   # each rank generates its shard

    latent_size = image_size // 8

    # ── Pre-load Inception v3 for IS (rank 0 only — used after gather) ───────
    from torchvision import models as tv_models
    if rank == 0:
        print("\nPre-loading Inception v3 for IS computation...")
        inception_for_is = tv_models.inception_v3(weights=tv_models.Inception_V3_Weights.DEFAULT)
        inception_for_is.aux_logits = False
        inception_for_is.eval().to(device)
    else:
        inception_for_is = None

    # ── Alpha sweep / scheduler ───────────────────────────────────────────────
    fid_results: dict = {}
    is_results: dict = {}
    scheduler_type = cfg["eval"].get("alpha_scheduler", None)
    alpha_fn = build_alpha_fn(scheduler_type)

    def _gather_images(local_images: torch.Tensor) -> torch.Tensor:
        """Gather per-rank image tensors to rank 0; returns full tensor on rank 0, None elsewhere."""
        if not is_dist:
            return local_images
        local_np = local_images.numpy()   # float32, already on CPU
        gather_list = [None] * world_size if rank == 0 else None
        dist.gather_object(local_np, gather_list, dst=0)
        if rank == 0:
            return torch.from_numpy(np.concatenate(gather_list, axis=0))
        return None

    if alpha_fn is not None:
        # ── Scheduler mode: single run ────────────────────────────────────────
        label = f"scheduler_{scheduler_type}"
        if rank == 0:
            print(f"\n{'─'*50}")
            print(f" alpha scheduler = {scheduler_type}")
            print(f"{'─'*50}")

        gen_images = generate_for_alpha(
            alpha_fn, model1, model2, sampler1, sampler2, blend_sampler, vae,
            rank_labels, cfg["sampling"], batch_size, latent_size, device,
        )
        gen_images = _gather_images(gen_images)

        if rank == 0:
            print("  Extracting Inception features for FID...")
            gen_features = get_inception_features(gen_images, batch_size=batch_size, device=device)
            fid = compute_fid(gen_features, ref_features)
            fid_results[label] = fid
            print(f"  FID = {fid:.4f}")

            print("  Extracting Inception probs for IS...")
            gen_probs = get_inception_probs(gen_images, batch_size=batch_size, device=device, model=inception_for_is)
            is_mean, is_std = compute_is_from_probs(gen_probs, splits=10)
            is_results[label] = {"mean": is_mean, "std": is_std}
            print(f"  IS  = {is_mean:.4f} ± {is_std:.4f}")

            grid_path = os.path.join(images_dir, f"grid_{label}.png")
            save_grid(gen_images, cfg["eval"]["grid_samples"], grid_path)

        if is_dist:
            dist.barrier()
    else:
        # ── Constant alpha sweep ──────────────────────────────────────────────
        alpha_list = cfg["eval"]["alpha_list"]

        for alpha in alpha_list:
            if rank == 0:
                print(f"\n{'─'*50}")
                print(f" alpha = {alpha:.2f}")
                print(f"{'─'*50}")

            gen_images = generate_for_alpha(
                alpha, model1, model2, sampler1, sampler2, blend_sampler, vae,
                rank_labels, cfg["sampling"], batch_size, latent_size, device,
            )
            gen_images = _gather_images(gen_images)

            if rank == 0:
                print("  Extracting Inception features for FID...")
                gen_features = get_inception_features(gen_images, batch_size=batch_size, device=device)
                fid = compute_fid(gen_features, ref_features)
                fid_results[alpha] = fid
                print(f"  FID = {fid:.4f}")

                print("  Extracting Inception probs for IS...")
                gen_probs = get_inception_probs(gen_images, batch_size=batch_size, device=device, model=inception_for_is)
                is_mean, is_std = compute_is_from_probs(gen_probs, splits=10)
                is_results[alpha] = {"mean": is_mean, "std": is_std}
                print(f"  IS  = {is_mean:.4f} ± {is_std:.4f}")

                alpha_str = f"{alpha:.2f}".replace(".", "_")
                grid_path = os.path.join(images_dir, f"grid_alpha_{alpha_str}.png")
                save_grid(gen_images, cfg["eval"]["grid_samples"], grid_path)

            if is_dist:
                dist.barrier()

    # ── Save results (rank 0 only) ────────────────────────────────────────────
    if rank == 0:
        json_path = os.path.join(exp_dir, "fid_results.json")
        with open(json_path, "w") as f:
            json.dump({str(k): v for k, v in sorted(fid_results.items())}, f, indent=2)
        print(f"\nSaved FID results → {json_path}")

        is_json_path = os.path.join(exp_dir, "is_results.json")
        with open(is_json_path, "w") as f:
            json.dump({str(k): v for k, v in sorted(is_results.items())}, f, indent=2)
        print(f"Saved IS results  → {is_json_path}")

        if alpha_fn is None:
            plot_fid_curve(fid_results, os.path.join(exp_dir, "fid_curve.png"))
            plot_is_curve(is_results, os.path.join(exp_dir, "is_curve.png"))
        print("\nDone.")

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
