"""
exp_velocity_rmse.py — Evaluate velocity prediction RMSE across timesteps.

For a pretrained velocity-prediction checkpoint:
  1. Sample a batch of real images from the dataset.
  2. Encode to latent x1 via VAE.
  3. At each timestep t, corrupt: x_t = t * x1 + (1 - t) * eps.
  4. Run the model to get predicted velocity v_pred.
  5. Compare to ground-truth velocity v_gt = x1 - eps.
  6. Compute per-sample RMSE, average over num_images_per_step images.
  7. Plot RMSE vs t and save to the experiment directory.

Usage:
    python exp_velocity_rmse.py --config configs/exp_velocity_rmse_config.yaml
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from diffusers.models import AutoencoderKL
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

_SIT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SIT_DIR not in sys.path:
    sys.path.insert(0, _SIT_DIR)

from models import SiT_models
from exp_model_validation import resolve_model_args, build_model, load_checkpoint_weights


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataset(data_path: str, image_size: int, num_images: int, seed: int):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: _center_crop(img, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = ImageFolder(data_path, transform=transform)
    n = min(num_images, len(dataset))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n, replace=False).tolist()
    return Subset(dataset, indices)


def _center_crop(pil_image, image_size):
    from PIL import Image
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    cy = (arr.shape[0] - image_size) // 2
    cx = (arr.shape[1] - image_size) // 2
    from PIL import Image
    return Image.fromarray(arr[cy:cy + image_size, cx:cx + image_size])


@torch.no_grad()
def evaluate_rmse_at_t(
    model,
    vae,
    loader: DataLoader,
    t_val: float,
    device: str,
) -> float:
    """Return mean per-sample RMSE of velocity prediction at a fixed timestep t."""
    rmse_list = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # Encode to latent space
        x1 = vae.encode(x).latent_dist.sample().mul_(0.18215)

        eps = torch.randn_like(x1)
        xt  = t_val * x1 + (1.0 - t_val) * eps

        t_tensor = torch.full((x1.shape[0],), t_val, device=device, dtype=x1.dtype)
        v_pred = model(xt, t_tensor, y=y)

        v_gt = x1 - eps

        # RMSE per sample: sqrt(mean over C,H,W of squared error)
        sq_err   = (v_pred - v_gt) ** 2
        per_sample_rmse = sq_err.flatten(1).mean(dim=1).sqrt()  # (B,)
        rmse_list.append(per_sample_rmse.cpu())

    return torch.cat(rmse_list).mean().item()


def plot_rmse_curve(time_steps: list, rmse_values: list, save_path: str, title: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(time_steps, rmse_values, marker="o", linewidth=2, color="steelblue")
    plt.xlabel("Timestep t")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved RMSE curve → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Velocity RMSE evaluation across timesteps")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    cli = parser.parse_args()

    cfg    = load_config(cli.config)
    device = cfg.get("device", cli.device)

    # ── Experiment directory ──────────────────────────────────────────────────
    exp_dir = os.path.join(
        cfg["experiment"]["output_dir"],
        cfg["experiment"]["name"],
    )
    os.makedirs(exp_dir, exist_ok=True)
    import shutil
    shutil.copy(cli.config, os.path.join(exp_dir, "config.yaml"))
    print(f"Experiment dir : {exp_dir}")

    # ── Load checkpoint & resolve model args ──────────────────────────────────
    ckpt_path = cfg["checkpoint"]["ckpt_path"]
    use_ema   = cfg["checkpoint"].get("use_ema", True)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "args" not in ckpt:
        raise KeyError("Checkpoint has no 'args' key.")
    model_args = resolve_model_args(ckpt["args"], cfg.get("model_overrides") or {})
    del ckpt

    assert model_args.prediction == "velocity", (
        f"This script only supports velocity-prediction checkpoints, "
        f"got prediction='{model_args.prediction}'"
    )
    print(f"  model={model_args.model}  image_size={model_args.image_size}"
          f"  num_classes={model_args.num_classes}  vae={model_args.vae}")

    # ── Build model & VAE ─────────────────────────────────────────────────────
    model = build_model(model_args, device)
    load_checkpoint_weights(model, ckpt_path, use_ema=use_ema)
    model.eval()

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{model_args.vae}").to(device)
    vae.eval()

    # ── Dataset ───────────────────────────────────────────────────────────────
    eval_cfg          = cfg["eval"]
    num_images        = int(eval_cfg["num_images_per_step"])
    batch_size        = int(eval_cfg["batch_size"])
    seed              = int(eval_cfg.get("seed", 0))
    time_steps        = [float(t) for t in eval_cfg["time_steps"]]
    num_workers       = int(cfg["data"].get("num_workers", 4))

    dataset = build_dataset(
        cfg["data"]["data_path"],
        model_args.image_size,
        num_images,
        seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Dataset: {len(dataset)} images  |  batch_size={batch_size}")

    # ── RMSE sweep over timesteps ─────────────────────────────────────────────
    rmse_values = []
    for t_val in time_steps:
        rmse = evaluate_rmse_at_t(model, vae, loader, t_val, device)
        rmse_values.append(rmse)
        print(f"  t={t_val:.2f}  RMSE={rmse:.6f}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {str(t): r for t, r in zip(time_steps, rmse_values)}
    json_path = os.path.join(exp_dir, "rmse_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results → {json_path}")

    title = cfg["experiment"]["name"].replace("_", " ")
    plot_rmse_curve(
        time_steps,
        rmse_values,
        save_path=os.path.join(exp_dir, "rmse_curve.png"),
        title=f"Velocity RMSE vs Timestep\n{title}",
    )


if __name__ == "__main__":
    main()
