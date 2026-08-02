"""
exp_model_validation.py — Validate SiT checkpoints: FID and IS vs training steps.

For a list of checkpoint steps, generates images (multi-GPU via torchrun or
single-GPU), computes FID and Inception Score, and plots training-progress curves.

Usage:
    # Multi-GPU (4 GPUs):
    torchrun --nproc_per_node=4 exp_model_validation.py \\
        --config configs/exp_model_validation_config.yaml

    # Single-GPU:
    python exp_model_validation.py \\
        --config configs/exp_model_validation_config.yaml --device cuda:0

Model architecture (model type, image_size, num_classes, vae, path_type,
prediction, etc.) is auto-loaded from ckpt["args"]. Use model_overrides in
the config to override any field if needed.
"""

import argparse
import json
import os
import shutil
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import yaml
from diffusers.models import AutoencoderKL
from scipy.linalg import sqrtm
from torchvision import datasets, models as tv_models, transforms

_SIT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SIT_DIR not in sys.path:
    sys.path.insert(0, _SIT_DIR)

from FID import get_inception_features
from IS import get_inception_probs, compute_is_from_probs
from models import SiT_models
from transport import Sampler, create_transport


# ─── Distributed setup ───────────────────────────────────────────────────────

def setup_dist(default_device: str = "cuda:0"):
    """
    Init NCCL process group if torchrun env vars (RANK, WORLD_SIZE) are present.
    Returns (rank, world_size, device_str).
    Falls back to single-GPU using default_device if not distributed.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = default_device
    return rank, world_size, device


def barrier(world_size: int):
    if world_size > 1:
        dist.barrier()


# ─── Config ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Model args ──────────────────────────────────────────────────────────────

def resolve_model_args(ckpt_args, overrides: dict) -> argparse.Namespace:
    """
    Merge checkpoint's args (argparse.Namespace) with optional user overrides.
    Non-null override values win; otherwise checkpoint values are used.
    Raises ValueError for required fields missing from both sources.
    """
    REQUIRED = ["model", "image_size", "num_classes", "vae", "path_type", "prediction"]
    OPTIONAL = ["loss_weight", "sample_eps", "train_eps", "latent_scale"]

    result = argparse.Namespace()
    for key in REQUIRED + OPTIONAL:
        override_val = (overrides or {}).get(key, None)
        if override_val is not None:
            setattr(result, key, override_val)
        else:
            ckpt_val = getattr(ckpt_args, key, None)
            if ckpt_val is None and key in REQUIRED:
                raise ValueError(
                    f"Required field '{key}' not found in checkpoint args and not specified "
                    f"in model_overrides. Please add it to the config."
                )
            setattr(result, key, ckpt_val)

    # Extra scale baked into the training latents (data.latent_scale in the
    # training config). Checkpoints written before it existed have no such field,
    # and 1.0 reproduces the original behaviour exactly.
    result.latent_scale = 1.0 if result.latent_scale is None else float(result.latent_scale)
    if result.latent_scale <= 0:
        raise ValueError(f"latent_scale must be > 0, got {result.latent_scale}")
    return result


def resolve_checkpoint_dir(checkpoint_dir: str) -> str:
    """Locate a checkpoint directory, tolerating train.py's `NNN-` name prefix.

    train.py prefixes every run directory with a running index that is not known
    when the validation config is written, so `results/<run-name>/checkpoints`
    also matches `results/053-<run-name>/checkpoints`. The match must be unique.
    """
    if os.path.isdir(checkpoint_dir):
        return checkpoint_dir

    norm = checkpoint_dir.rstrip("/")
    leaf = os.path.basename(norm)                    # "checkpoints"
    run_dir = os.path.dirname(norm)                  # "results/<run-name>"
    run_name = os.path.basename(run_dir)
    results_dir = os.path.dirname(run_dir) or "."
    if not run_name:
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    matches = [
        os.path.join(m, leaf) for m in sorted(glob(os.path.join(results_dir, f"*{run_name}")))
        if os.path.isdir(os.path.join(m, leaf))
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Checkpoint directory '{checkpoint_dir}' is ambiguous — "
            f"{len(matches)} runs match: {matches}. Use the full path in the config."
        )
    raise FileNotFoundError(
        f"Checkpoint directory not found: {checkpoint_dir} "
        f"(also tried '{os.path.join(results_dir, '*' + run_name)}')"
    )


# ─── Model / VAE / Transport ─────────────────────────────────────────────────

def build_model(model_args: argparse.Namespace, device: str):
    """Instantiate a SiT model (no weights loaded yet)."""
    latent_size = model_args.image_size // 8
    model = SiT_models[model_args.model](
        input_size=latent_size,
        num_classes=model_args.num_classes,
    ).to(device)
    model.eval()
    return model


def load_checkpoint_weights(model, ckpt_path: str, use_ema: bool) -> None:
    """Load EMA or plain weights from a checkpoint file into an already-built model."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    key = "ema" if use_ema else "model"
    if key not in state:
        raise KeyError(
            f"Key '{key}' not found in checkpoint {ckpt_path}. "
            f"Available keys: {list(state.keys())}"
        )
    model.load_state_dict(state[key])
    model.eval()


def build_transport_and_sampler(model_args: argparse.Namespace):
    """Build Transport and Sampler from resolved model args."""
    transport = create_transport(
        path_type=getattr(model_args, "path_type", "Linear"),
        prediction=getattr(model_args, "prediction", "velocity"),
        loss_weight=getattr(model_args, "loss_weight", None),
        train_eps=getattr(model_args, "train_eps", None),
        sample_eps=getattr(model_args, "sample_eps", None),
        t_min=0.0,
    )
    return transport, Sampler(transport)


def load_vae(vae_name: str, device: str):
    """Load frozen AutoencoderKL."""
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_name}").to(device)
    vae.eval()
    return vae


# ─── Reference features ──────────────────────────────────────────────────────

def get_reference_features(
    ref_data_path: str,
    num_ref: int,
    image_size: int,
    batch_size: int,
    device: str,
    seed: int = 0,
) -> np.ndarray:
    """Load reference images (ImageFolder-compatible) and extract Inception features (rank 0 only)."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(ref_data_path, transform=transform)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(num_ref, len(dataset)), replace=False)
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    imgs = torch.cat([x for x, _ in loader], dim=0)
    print(f"  Extracting Inception features for {len(imgs)} reference images...")
    return get_inception_features(imgs, batch_size=batch_size, device=device)


# ─── Class labels ─────────────────────────────────────────────────────────────

def make_balanced_labels(num_images: int, num_classes: int, seed: int = 0) -> np.ndarray:
    """Return (num_images,) int64 labels balanced across all classes."""
    base = num_images // num_classes
    remainder = num_images % num_classes
    labels = []
    for c in range(num_classes):
        labels.extend([c] * (base + (1 if c < remainder else 0)))
    labels = np.array(labels, dtype=np.int64)
    np.random.default_rng(seed).shuffle(labels)
    return labels


# ─── Image generation ─────────────────────────────────────────────────────────

@torch.no_grad()
def generate_images_rank(
    model,
    sampler: Sampler,
    vae,
    labels_this_rank: np.ndarray,
    sampling_cfg: dict,
    batch_size: int,
    latent_size: int,
    num_classes: int,
    device: str,
    seed: int,
    rank: int,
    world_size: int,
    latent_scale: float = 1.0,
) -> np.ndarray:
    """
    Generate this rank's share of images in memory.

    Returns uint8 numpy array of shape (N, 3, H, W) — no disk I/O.
    Per-rank seed: seed * world_size + rank, ensuring different noise per GPU
    while keeping the same noise across checkpoint evaluations for fair comparison.
    """
    torch.manual_seed(seed * world_size + rank)

    n         = len(labels_this_rank)
    num_steps = sampling_cfg["num_sampling_steps"]
    method    = sampling_cfg["sampling_method"]
    atol      = sampling_cfg["atol"]
    rtol      = sampling_cfg["rtol"]
    cfg_scale = sampling_cfg["cfg_scale"]
    use_cfg   = cfg_scale > 1.0

    sample_fn = sampler.sample_ode(
        sampling_method=method, num_steps=num_steps, atol=atol, rtol=rtol
    )

    batches = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bsz = end - start
        y = torch.tensor(labels_this_rank[start:end], device=device)
        z = torch.randn(bsz, 4, latent_size, latent_size, device=device)

        if use_cfg:
            z      = torch.cat([z, z], dim=0)
            y_null = torch.full((bsz,), num_classes, dtype=torch.long, device=device)
            y      = torch.cat([y, y_null], dim=0)
            model_kwargs  = dict(y=y, cfg_scale=cfg_scale)
            model_fn      = model.forward_with_cfg
        else:
            model_kwargs  = dict(y=y)
            model_fn      = model.forward

        samples = sample_fn(z, model_fn, **model_kwargs)[-1]

        if use_cfg:
            samples, _ = samples.chunk(2, dim=0)

        # Undo both the SD-VAE constant and any extra scale baked into the
        # training latents, otherwise the decoded images are mis-scaled.
        decoded = vae.decode(samples / (0.18215 * latent_scale)).sample
        decoded = (decoded.clamp(-1.0, 1.0) + 1.0) / 2.0  # float32 [0, 1]

        # Convert to uint8 CPU immediately to free GPU memory
        uint8 = (decoded * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        batches.append(uint8)

        if rank == 0:
            print(f"\r  Generated {end}/{n} images (rank 0)", end="", flush=True)

    if rank == 0:
        print()

    return np.concatenate(batches, axis=0)  # (N, 3, H, W) uint8


def gather_images(
    local_arr: np.ndarray,
    rank: int,
    world_size: int,
) -> "np.ndarray | None":
    """
    Gather uint8 numpy arrays from all ranks to rank 0 (in memory, no disk I/O).

    Uses dist.gather_object (pickle-based, handles variable-length arrays).
    Rank 0 returns concatenated (N_total, 3, H, W) array; other ranks return None.
    If world_size == 1, returns local_arr directly without any distributed call.
    """
    if world_size == 1:
        return local_arr

    gather_list = [None] * world_size if rank == 0 else None
    dist.gather_object(local_arr, gather_list, dst=0)

    if rank == 0:
        return np.concatenate(gather_list, axis=0)
    return None


# ─── FID / IS helpers ────────────────────────────────────────────────────────

def compute_fid_from_features(gen_features: np.ndarray, ref_features: np.ndarray) -> float:
    """Fréchet Inception Distance from pre-extracted Inception feature arrays."""
    mu_r, mu_g  = ref_features.mean(axis=0), gen_features.mean(axis=0)
    sigma_r     = np.cov(ref_features, rowvar=False)
    sigma_g     = np.cov(gen_features, rowvar=False)
    covmean, _  = sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_r - mu_g
    return float(diff @ diff + np.trace(sigma_r + sigma_g - 2.0 * covmean))


def extract_features_chunked(
    uint8_arr: np.ndarray,
    fid_batch_size: int,
    chunk_size: int,
    device: str,
    inception_model=None,
    mode: str = "fid",
) -> np.ndarray:
    """
    Process a uint8 numpy array (N, 3, H, W) in float32 chunks to avoid
    creating a 39 GB float32 tensor when N=50k at 256x256.

    mode='fid' : calls get_inception_features → returns (N, 2048) features
    mode='is'  : calls get_inception_probs    → returns (N, 1000) probs
                 (pass inception_model to avoid reloading per chunk)
    """
    results = []
    n = len(uint8_arr)
    for start in range(0, n, chunk_size):
        end   = min(start + chunk_size, n)
        chunk = torch.from_numpy(uint8_arr[start:end]).float() / 255.0  # [0, 1]
        if mode == "fid":
            feat = get_inception_features(chunk, batch_size=fid_batch_size, device=device)
        else:
            feat = get_inception_probs(
                chunk, batch_size=fid_batch_size, device=device, model=inception_model
            )
        results.append(feat)
        del chunk
        print(f"\r  [{mode.upper()}] processed {end}/{n}", end="", flush=True)
    print()
    return np.concatenate(results, axis=0)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_fid_curve(results: dict, save_path: str) -> None:
    steps = sorted(results.keys())
    fids  = [results[s]["fid"] for s in steps]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, fids, marker="o", linewidth=2)
    plt.xlabel("Training Steps")
    plt.ylabel("FID")
    plt.title("FID vs Training Steps")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved FID curve → {save_path}")


def plot_is_curve(results: dict, save_path: str) -> None:
    steps = sorted(results.keys())
    means = [results[s]["is_mean"] for s in steps]
    stds  = [results[s]["is_std"]  for s in steps]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, means, marker="o", linewidth=2)
    plt.fill_between(
        steps,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.2,
    )
    plt.xlabel("Training Steps")
    plt.ylabel("Inception Score")
    plt.title("IS vs Training Steps")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved IS curve  → {save_path}")


# ─── Results I/O ─────────────────────────────────────────────────────────────

def save_results_json(results: dict, path: str) -> None:
    """Save results dict to JSON; casts numpy floats to Python float."""
    serializable = {
        str(step): {
            "fid":     float(v["fid"]),
            "is_mean": float(v["is_mean"]),
            "is_std":  float(v["is_std"]),
        }
        for step, v in sorted(results.items())
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate SiT checkpoints: FID + IS vs training steps"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for single-GPU mode (overrides config, e.g. cuda:1)")
    cli_args = parser.parse_args()

    cfg = load_config(cli_args.config)

    # ── 1. Distributed setup ─────────────────────────────────────────────────
    default_device = cli_args.device or cfg.get("device", "cuda:0")
    rank, world_size, device = setup_dist(default_device)

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── 2. Experiment directory (rank 0 creates it) ───────────────────────────
    exp_dir = os.path.join(
        cfg["experiment"]["experiments_base_dir"],
        cfg["experiment"]["name"],
    )
    if rank == 0:
        os.makedirs(exp_dir, exist_ok=True)
        shutil.copy(cli_args.config, os.path.join(exp_dir, "config.yaml"))
        print(f"Experiment dir : {exp_dir}")
        print(f"Rank {rank}/{world_size} on {device}")
    barrier(world_size)

    # ── 3. Checkpoint config ─────────────────────────────────────────────────
    ckpt_dir     = resolve_checkpoint_dir(cfg["checkpoint"]["checkpoint_dir"])
    steps        = cfg["checkpoint"]["steps"]
    if rank == 0 and ckpt_dir != cfg["checkpoint"]["checkpoint_dir"]:
        print(f"Checkpoints    : {ckpt_dir}")
    use_ema      = cfg["checkpoint"].get("use_ema", True)
    skip_missing = cfg["checkpoint"].get("skip_missing", False)

    assert cfg["eval"]["num_images"] >= 10, \
        "eval.num_images must be >= 10 (IS splits=10 requires at least 10 images)"

    # ── 4. Extract model args from first available checkpoint ─────────────────
    first_ckpt_path = next(
        (os.path.join(ckpt_dir, f"{s:07d}.pt") for s in steps
         if os.path.isfile(os.path.join(ckpt_dir, f"{s:07d}.pt"))),
        None,
    )
    if first_ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint files found in: {ckpt_dir}")
    if rank == 0:
        print(f"\nExtracting model architecture from: {first_ckpt_path}")
    first_ckpt = torch.load(first_ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = first_ckpt.get("args", None)
    model_args = resolve_model_args(ckpt_args, cfg.get("model_overrides") or {})
    del first_ckpt

    if rank == 0:
        print(f"  model={model_args.model}, image_size={model_args.image_size}, "
              f"num_classes={model_args.num_classes}")
        print(f"  vae={model_args.vae}, path_type={model_args.path_type}, "
              f"prediction={model_args.prediction}")
        print(f"  latent_scale={model_args.latent_scale} "
              f"(generated latents are divided by 0.18215 * latent_scale before decoding)")

    # ── 5. Build model, VAE, transport+sampler (all ranks) ───────────────────
    if rank == 0:
        print("\nBuilding model...")
    model = build_model(model_args, device)

    if rank == 0:
        print(f"Loading VAE (sd-vae-ft-{model_args.vae})...")
    vae = load_vae(model_args.vae, device)

    _transport, sampler = build_transport_and_sampler(model_args)
    latent_size = model_args.image_size // 8

    # ── 6. Reference features + Inception v3 for IS (rank 0 only) ────────────
    if rank == 0:
        ref_data_path = cfg["data"].get("ref_data_path") or cfg["data"]["imagenet_val_path"]
        print(f"\nBuilding reference features from {ref_data_path} ...")
        ref_features = get_reference_features(
            ref_data_path,
            cfg["data"]["num_ref_images"],
            model_args.image_size,
            cfg["eval"]["fid_batch_size"],
            device,
            seed=cfg["eval"]["seed"],
        )
        print("Pre-loading Inception v3 for IS...")
        inception_for_is = tv_models.inception_v3(
            weights=tv_models.Inception_V3_Weights.DEFAULT
        )
        inception_for_is.aux_logits = False
        inception_for_is.eval().to(device)
    else:
        ref_features     = None
        inception_for_is = None
    barrier(world_size)

    # ── 7. Class labels split per rank ───────────────────────────────────────
    num_images  = cfg["eval"]["num_images"]
    seed        = cfg["eval"]["seed"]
    all_labels  = make_balanced_labels(num_images, model_args.num_classes, seed=seed)
    labels_this_rank = all_labels[rank::world_size]

    # ── 8. Parse cfg_scale — scalar or list ──────────────────────────────────
    cfg_scales_raw = cfg["sampling"]["cfg_scale"]
    cfg_scales  = cfg_scales_raw if isinstance(cfg_scales_raw, list) else [cfg_scales_raw]
    use_subdirs = len(cfg_scales) > 1

    # ── 9. CFG sweep (outer) + checkpoint loop (inner) ───────────────────────
    for cfg_scale_val in cfg_scales:

        # Output directory: per-cfg subdir when sweeping, flat otherwise
        if use_subdirs:
            out_dir = os.path.join(exp_dir, f"cfg_{cfg_scale_val}")
        else:
            out_dir = exp_dir

        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)
            print(f"\n{'═'*55}")
            print(f" cfg_scale = {cfg_scale_val}")
            print(f"{'═'*55}")
        barrier(world_size)

        # Per-cfg sampling config (only cfg_scale overridden)
        sampling_cfg = dict(cfg["sampling"])
        sampling_cfg["cfg_scale"] = cfg_scale_val

        # Per-cfg resume: load existing results.json if present
        results_json_path = os.path.join(out_dir, "results.json")
        results: dict = {}
        if rank == 0 and os.path.isfile(results_json_path):
            with open(results_json_path) as f:
                loaded = json.load(f)
            results = {int(k): v for k, v in loaded.items()}
            if results:
                print(f"Resuming: {len(results)} step(s) already completed: "
                      f"{sorted(results.keys())}")

        for step in steps:
            ckpt_path = os.path.join(ckpt_dir, f"{step:07d}.pt")

            # Broadcast skip decision from rank 0 so all ranks agree
            if rank == 0:
                skip_step = step in results
            else:
                skip_step = False
            if world_size > 1:
                skip_tensor = torch.tensor([int(skip_step)], device=device)
                dist.broadcast(skip_tensor, src=0)
                skip_step = bool(skip_tensor.item())

            if skip_step:
                if rank == 0:
                    print(f"Step {step:,d} already done — skipping.")
                continue

            # Missing checkpoint
            if not os.path.isfile(ckpt_path):
                if skip_missing:
                    if rank == 0:
                        print(f"WARNING: checkpoint not found: {ckpt_path} — skipping.")
                    continue
                else:
                    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            if rank == 0:
                print(f"\n{'─'*55}")
                print(f" Step {step:,d}  →  {ckpt_path}")
                print(f"{'─'*55}")

            # Load weights, verify architecture consistency
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "args" in state:
                step_args = resolve_model_args(state["args"], cfg.get("model_overrides") or {})
                if (step_args.model != model_args.model
                        or step_args.image_size != model_args.image_size
                        or step_args.num_classes != model_args.num_classes):
                    raise ValueError(
                        f"Architecture mismatch at step {step}: "
                        f"expected model={model_args.model} image_size={model_args.image_size}, "
                        f"got model={step_args.model} image_size={step_args.image_size}. "
                        f"All steps must share the same architecture."
                    )
            key = "ema" if use_ema else "model"
            model.load_state_dict(state[key])
            model.eval()
            del state

            # Generate images (all ranks)
            if rank == 0:
                print("  Generating images...")
            local_arr = generate_images_rank(
                model, sampler, vae,
                labels_this_rank,
                sampling_cfg,
                cfg["eval"]["batch_size"],
                latent_size,
                model_args.num_classes,
                device,
                seed, rank, world_size,
                latent_scale=model_args.latent_scale,
            )

            # Gather to rank 0 (in-memory, no disk I/O)
            all_images = gather_images(local_arr, rank, world_size)
            del local_arr
            barrier(world_size)

            # FID + IS on rank 0 only
            if rank == 0:
                all_images = all_images[:num_images]  # trim any excess from uneven split
                fid_batch  = cfg["eval"]["fid_batch_size"]
                chunk_size = cfg["eval"]["fid_chunk_size"]

                print("  Extracting Inception features for FID...")
                gen_features = extract_features_chunked(
                    all_images, fid_batch, chunk_size, device, mode="fid"
                )
                fid = compute_fid_from_features(gen_features, ref_features)
                print(f"  FID = {fid:.4f}")

                print("  Computing Inception Score...")
                gen_probs = extract_features_chunked(
                    all_images, fid_batch, chunk_size, device,
                    inception_model=inception_for_is, mode="is",
                )
                is_mean, is_std = compute_is_from_probs(gen_probs, splits=10)
                print(f"  IS  = {is_mean:.4f} ± {is_std:.4f}")

                del all_images, gen_features, gen_probs

                results[step] = {
                    "fid":     float(fid),
                    "is_mean": float(is_mean),
                    "is_std":  float(is_std),
                }
                save_results_json(results, results_json_path)
                print(f"  Saved → {results_json_path}")

            barrier(world_size)

        # ── Final save + plots for this cfg ──────────────────────────────────
        if rank == 0:
            save_results_json(results, results_json_path)
            print(f"\nFinal results saved → {results_json_path}")
            if results:
                plot_fid_curve(results, os.path.join(out_dir, "fid_curve.png"))
                plot_is_curve(results,  os.path.join(out_dir, "is_curve.png"))
        barrier(world_size)

    # ── Done ─────────────────────────────────────────────────────────────────
    if rank == 0:
        print("\nAll cfg_scale evaluations complete.")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
