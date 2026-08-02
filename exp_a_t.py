"""Estimate the frozen teacher flow-map sensitivity curve a(t).

For a linear interpolant ``z_t = (1 - t) z_0 + t z_1``, this script estimates

    a(t) = E_{z_t,q}[||Phi(1, t; z_t) q||_2^2],

where ``q`` is a unit-norm Rademacher probe and ``Phi`` is the derivative of
the teacher ODE flow map with respect to its input state.  No Jacobian, JVP, or
backward pass is used.  Instead, ``Phi q`` is estimated with a central finite
difference of two ordinary, frozen-teacher Heun rollouts.

The same selected data latents, Gaussian source latents, labels, and probes are
reused at every time point.  The output directory contains per-sample targets
(``a_t_targets.npz``), aggregate statistics (JSON and CSV), and a curve plot.

Usage:
    python exp_a_t.py --config configs/exp_a_t_config_XL_velocity_imagenet.yaml
"""

import argparse
import bisect
import csv
import json
import math
import os
import shutil
from contextlib import nullcontext
from glob import glob
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from models import SiT_models


LATENT_SCALE = 0.18215


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    for section in ("experiment", "checkpoint", "model_overrides", "data", "eval"):
        if section not in config:
            raise ValueError(f"Missing required config section '{section}': {path}")
    return config


def resolve_checkpoint_path(checkpoint_cfg: dict) -> str:
    """Resolve an exact checkpoint, including train.py's numeric run prefix."""
    ckpt_path = checkpoint_cfg.get("ckpt_path")
    if ckpt_path is None:
        checkpoint_dir = checkpoint_cfg.get("checkpoint_dir")
        step = checkpoint_cfg.get("step")
        if checkpoint_dir is None or step is None:
            raise ValueError(
                "checkpoint must define ckpt_path, or both checkpoint_dir and step"
            )
        ckpt_path = os.path.join(checkpoint_dir, f"{int(step):07d}.pt")

    if os.path.isfile(ckpt_path):
        return ckpt_path

    norm = os.path.normpath(ckpt_path)
    checkpoint_dir = os.path.dirname(norm)
    run_dir = os.path.dirname(checkpoint_dir)
    results_dir = os.path.dirname(run_dir) or "."
    run_name = os.path.basename(run_dir)
    filename = os.path.basename(norm)
    matches = sorted(
        p
        for p in glob(os.path.join(results_dir, f"*{run_name}", "checkpoints", filename))
        if os.path.isfile(p)
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Checkpoint path is ambiguous; matches: {matches}")
    raise FileNotFoundError(
        f"Checkpoint not found: {ckpt_path} (also searched for a numeric run prefix)"
    )


def _value_from_args(ckpt_args, key: str):
    if isinstance(ckpt_args, dict):
        return ckpt_args.get(key)
    return getattr(ckpt_args, key, None)


def resolve_model_args(ckpt_args, overrides: dict) -> SimpleNamespace:
    required = ("model", "image_size", "num_classes", "vae", "path_type", "prediction")
    values = {}
    for key in required:
        value = overrides.get(key) if overrides else None
        if value is None and ckpt_args is not None:
            value = _value_from_args(ckpt_args, key)
        if value is None:
            raise ValueError(
                f"Model field '{key}' is absent from both checkpoint args and model_overrides"
            )
        values[key] = value

    latent_scale = (overrides or {}).get("latent_scale")
    if latent_scale is None and ckpt_args is not None:
        latent_scale = _value_from_args(ckpt_args, "latent_scale")
    values["latent_scale"] = 1.0 if latent_scale is None else float(latent_scale)
    values["image_size"] = int(values["image_size"])
    values["num_classes"] = int(values["num_classes"])
    if values["latent_scale"] <= 0:
        raise ValueError("latent_scale must be positive")
    if values["path_type"] != "Linear":
        raise ValueError("This a(t) experiment currently requires path_type='Linear'")
    if values["prediction"] != "velocity":
        raise ValueError("This a(t) experiment requires a velocity-prediction teacher")
    return SimpleNamespace(**values)


class PackedPosteriorDataset(Dataset):
    """Read deterministic posterior parameters from packed ImageNet latents."""

    def __init__(self, root: str, orientation: int = 0):
        if orientation not in (0, 1):
            raise ValueError("data.orientation must be 0 (original) or 1 (flipped)")
        filenames = sorted(f for f in os.listdir(root) if f.endswith(".npy"))
        if not filenames:
            raise FileNotFoundError(f"No packed .npy latent files found in {root}")

        self.orientation = orientation
        self.entries: List[Tuple[str, int]] = []
        self.cumulative_sizes: List[int] = []
        self._mmaps: Dict[str, np.ndarray] = {}
        total = 0
        expected_shape = None
        for label, filename in enumerate(filenames):
            path = os.path.join(root, filename)
            array = np.load(path, mmap_mode="r")
            if array.ndim != 6 or array.shape[1:3] != (2, 2):
                raise ValueError(
                    f"Unexpected packed latent shape {array.shape} in {path}; "
                    "expected (N, 2, 2, C, H, W)"
                )
            latent_shape = tuple(array.shape[3:])
            if expected_shape is None:
                expected_shape = latent_shape
            elif latent_shape != expected_shape:
                raise ValueError(
                    f"Inconsistent latent shape {latent_shape} in {path}; "
                    f"expected {expected_shape}"
                )
            total += int(array.shape[0])
            self.entries.append((path, label))
            self.cumulative_sizes.append(total)
            del array
        self.total = total

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int):
        if index < 0 or index >= self.total:
            raise IndexError(index)
        class_pos = bisect.bisect_right(self.cumulative_sizes, index)
        previous = self.cumulative_sizes[class_pos - 1] if class_pos else 0
        local_index = index - previous
        path, label = self.entries[class_pos]
        if path not in self._mmaps:
            self._mmaps[path] = np.load(path, mmap_mode="r")
        sample = self._mmaps[path][local_index, self.orientation]
        mean = torch.from_numpy(sample[0].copy())
        std = torch.from_numpy(sample[1].copy())
        return mean, std, label


class CenterCropTransform:
    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, pil_image):
        from PIL import Image

        resampling = getattr(Image, "Resampling", Image)
        while min(*pil_image.size) >= 2 * self.image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=resampling.BOX
            )
        scale = self.image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size),
            resample=resampling.BICUBIC,
        )
        array = np.asarray(pil_image)
        crop_y = (array.shape[0] - self.image_size) // 2
        crop_x = (array.shape[1] - self.image_size) // 2
        return Image.fromarray(
            array[crop_y:crop_y + self.image_size, crop_x:crop_x + self.image_size]
        )


def select_indices(dataset_size: int, num_samples: int, seed: int) -> np.ndarray:
    if num_samples <= 0:
        raise ValueError("eval.num_samples must be positive")
    if num_samples > dataset_size:
        raise ValueError(
            f"Requested {num_samples} samples, but the dataset only has {dataset_size}"
        )
    return np.random.default_rng(seed).choice(
        dataset_size, size=num_samples, replace=False
    ).astype(np.int64)


def collect_packed_latents(
    root: str,
    num_samples: int,
    batch_size: int,
    num_workers: int,
    orientation: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    dataset = PackedPosteriorDataset(root, orientation=orientation)
    indices = select_indices(len(dataset), num_samples, seed)
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    latents, labels = [], []
    for mean, std, label in loader:
        posterior_noise = torch.randn(std.shape, generator=generator, dtype=torch.float32)
        # Any data.latent_scale is already baked into packed posterior files,
        # exactly as assumed by train.py's PackedLatentImageFolder.
        latents.append((mean.float() + std.float() * posterior_noise) * LATENT_SCALE)
        labels.append(label.long())
    return torch.cat(latents), torch.cat(labels), indices


@torch.inference_mode()
def collect_raw_image_latents(
    root: str,
    model_args: SimpleNamespace,
    num_samples: int,
    batch_size: int,
    num_workers: int,
    latent_scale: float,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    from diffusers.models import AutoencoderKL
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    transform = transforms.Compose([
        CenterCropTransform(model_args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = ImageFolder(root, transform=transform)
    indices = select_indices(len(dataset), num_samples, seed)
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{model_args.vae}"
    ).to(device)
    vae.eval()
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    latents, labels = [], []
    for images, label in loader:
        posterior = vae.encode(images.to(device, non_blocking=True)).latent_dist
        noise = torch.randn(
            posterior.std.shape,
            generator=generator,
            device=device,
            dtype=posterior.std.dtype,
        )
        z1 = (posterior.mean + posterior.std * noise) * LATENT_SCALE * latent_scale
        latents.append(z1.float().cpu())
        labels.append(label.long())
    del vae
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(latents), torch.cat(labels), indices


def collect_fixed_samples(
    data_cfg: dict,
    eval_cfg: dict,
    model_args: SimpleNamespace,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, str]:
    num_samples = int(eval_cfg["num_samples"])
    batch_size = int(data_cfg.get("loading_batch_size", eval_cfg["batch_size"]))
    num_workers = int(data_cfg.get("num_workers", 4))
    seed = int(eval_cfg.get("seed", 0))
    latent_scale_cfg = data_cfg.get("latent_scale")
    latent_scale = (
        model_args.latent_scale if latent_scale_cfg is None else float(latent_scale_cfg)
    )
    if latent_scale <= 0:
        raise ValueError("data.latent_scale must be positive")

    packed_path = data_cfg.get("packed_latent_data_path")
    if packed_path and os.path.isdir(packed_path):
        z1, labels, indices = collect_packed_latents(
            packed_path,
            num_samples,
            batch_size,
            num_workers,
            int(data_cfg.get("orientation", 0)),
            seed,
        )
        return z1, labels, indices, packed_path

    raw_path = data_cfg.get("data_path")
    if raw_path and os.path.isdir(raw_path):
        z1, labels, indices = collect_raw_image_latents(
            raw_path,
            model_args,
            num_samples,
            batch_size,
            num_workers,
            latent_scale,
            seed,
            device,
        )
        return z1, labels, indices, raw_path

    raise FileNotFoundError(
        "Neither data.packed_latent_data_path nor data.data_path exists. "
        f"Got packed={packed_path!r}, raw={raw_path!r}"
    )


def make_fixed_source_and_probes(
    z1: torch.Tensor, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed + 2)
    z0 = torch.randn(z1.shape, generator=generator, dtype=torch.float32)
    signs = torch.randint(0, 2, z1.shape, generator=generator, dtype=torch.int8)
    dimension = int(np.prod(z1.shape[1:]))
    q = signs.to(torch.float32).mul_(2).sub_(1).div_(math.sqrt(dimension))
    return z0, q


def parse_compute_dtype(name: str) -> Optional[torch.dtype]:
    normalized = name.lower().replace("torch.", "")
    choices = {
        "float32": None,
        "fp32": None,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if normalized not in choices:
        raise ValueError(
            f"Unsupported eval.compute_dtype={name!r}; choose float32, bfloat16, or float16"
        )
    return choices[normalized]


def _autocast_context(device: torch.device, compute_dtype: Optional[torch.dtype]):
    if compute_dtype is None:
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("float16/bfloat16 compute_dtype is only supported on CUDA")
    return torch.autocast(device_type="cuda", dtype=compute_dtype)


@torch.inference_mode()
def teacher_flow_heun(
    teacher: torch.nn.Module,
    z_start: torch.Tensor,
    t_start: float,
    labels: torch.Tensor,
    base_steps: int,
    compute_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Integrate dz/dt = teacher(z,t,y) from t_start to 1 with Heun."""
    if not 0.0 <= t_start <= 1.0:
        raise ValueError(f"t_start must be in [0, 1], got {t_start}")
    if base_steps <= 0:
        raise ValueError(f"base_steps must be positive, got {base_steps}")
    if t_start == 1.0:
        return z_start.float()

    num_steps = max(1, math.ceil(base_steps * (1.0 - t_start)))
    step_size = (1.0 - t_start) / num_steps
    z = z_start.float()
    device = z.device

    for step in range(num_steps):
        time_value = t_start + step * step_size
        time_1 = torch.full(
            (z.shape[0],), time_value, device=device, dtype=torch.float32
        )
        with _autocast_context(device, compute_dtype):
            velocity_1 = teacher(z, time_1, y=labels)
        velocity_1 = velocity_1.float()
        z_predict = z + step_size * velocity_1

        time_2 = torch.full(
            (z.shape[0],), time_value + step_size, device=device, dtype=torch.float32
        )
        with _autocast_context(device, compute_dtype):
            velocity_2 = teacher(z_predict, time_2, y=labels)
        z = z + 0.5 * step_size * (velocity_1 + velocity_2.float())
    return z


@torch.inference_mode()
def finite_difference_targets(
    teacher: torch.nn.Module,
    z_t: torch.Tensor,
    q: torch.Tensor,
    t_start: float,
    labels: torch.Tensor,
    eta: float,
    base_steps: int,
    compute_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    """Return per-sample ||(F(z+eps q)-F(z-eps q))/(2 eps)||^2."""
    if eta <= 0:
        raise ValueError(f"eta must be positive, got {eta}")
    z_norm = z_t.flatten(1).norm(dim=1)
    if torch.any(z_norm == 0):
        raise ValueError("Encountered a zero-norm z_t; relative epsilon is undefined")
    epsilon = eta * z_norm
    epsilon_view = epsilon.reshape(z_t.shape[0], *([1] * (z_t.ndim - 1)))
    z_pair = torch.cat((z_t + epsilon_view * q, z_t - epsilon_view * q), dim=0)
    label_pair = torch.cat((labels, labels), dim=0)
    endpoint_pair = teacher_flow_heun(
        teacher,
        z_pair,
        t_start,
        label_pair,
        base_steps,
        compute_dtype,
    )
    batch_size = z_t.shape[0]
    phi_q = (
        endpoint_pair[:batch_size] - endpoint_pair[batch_size:]
    ) / (2.0 * epsilon_view)
    return phi_q.flatten(1).double().square().sum(dim=1).cpu()


@torch.inference_mode()
def evaluate_at_time(
    teacher: torch.nn.Module,
    z0_cpu: torch.Tensor,
    z1_cpu: torch.Tensor,
    q_cpu: torch.Tensor,
    labels_cpu: torch.Tensor,
    t_value: float,
    batch_size: int,
    eta: float,
    base_steps: int,
    compute_dtype: Optional[torch.dtype],
    device: torch.device,
    limit: Optional[int] = None,
) -> torch.Tensor:
    sample_count = len(z1_cpu) if limit is None else min(int(limit), len(z1_cpu))
    targets = []
    for start in range(0, sample_count, batch_size):
        end = min(start + batch_size, sample_count)
        z0 = z0_cpu[start:end].to(device, non_blocking=True)
        z1 = z1_cpu[start:end].to(device, non_blocking=True)
        q = q_cpu[start:end].to(device, non_blocking=True)
        labels = labels_cpu[start:end].to(device, non_blocking=True)
        z_t = (1.0 - t_value) * z0 + t_value * z1
        targets.append(
            finite_difference_targets(
                teacher,
                z_t,
                q,
                t_value,
                labels,
                eta,
                base_steps,
                compute_dtype,
            )
        )
    return torch.cat(targets)


def summarize_targets(targets: torch.Tensor) -> dict:
    values = targets.double()
    count = int(values.numel())
    std = values.std(unbiased=True).item() if count > 1 else 0.0
    return {
        "count": count,
        "mean": values.mean().item(),
        "std": std,
        "standard_error": std / math.sqrt(count),
        "median": torch.quantile(values, 0.5).item(),
        "q10": torch.quantile(values, 0.1).item(),
        "q90": torch.quantile(values, 0.9).item(),
        "min": values.min().item(),
        "max": values.max().item(),
    }


def build_time_grid(eval_cfg: dict) -> List[float]:
    if eval_cfg.get("time_steps") is not None:
        times = [float(t) for t in eval_cfg["time_steps"]]
    else:
        num_midpoints = int(eval_cfg.get("num_midpoints", 16))
        if num_midpoints <= 0:
            raise ValueError("eval.num_midpoints must be positive")
        times = [(j + 0.5) / num_midpoints for j in range(num_midpoints)]
    if eval_cfg.get("include_t1", True) and not any(t == 1.0 for t in times):
        times.append(1.0)
    if any(t < 0.0 or t > 1.0 for t in times):
        raise ValueError(f"All time points must lie in [0,1], got {times}")
    if len(set(times)) != len(times):
        raise ValueError(f"Duplicate time points are not allowed: {times}")
    return times


def run_stability_checks(
    teacher: torch.nn.Module,
    z0: torch.Tensor,
    z1: torch.Tensor,
    q: torch.Tensor,
    labels: torch.Tensor,
    cfg: dict,
    eval_cfg: dict,
    compute_dtype: Optional[torch.dtype],
    device: torch.device,
) -> dict:
    if not cfg.get("enabled", False):
        return {"enabled": False}

    times = [float(t) for t in cfg.get("time_steps", [0.09375, 0.46875, 0.90625])]
    eta_values = [float(x) for x in cfg.get("eta_values", [1e-3, 3e-3, 1e-2])]
    step_values = [int(x) for x in cfg.get("base_steps_values", [32, 64])]
    limit = int(cfg.get("num_samples", min(32, len(z1))))
    batch_size = int(cfg.get("batch_size", eval_cfg["batch_size"]))
    main_eta = float(eval_cfg.get("eta", 3e-3))
    main_steps = int(eval_cfg.get("base_steps", 32))
    if not times or any(t < 0.0 or t > 1.0 for t in times):
        raise ValueError(f"stability.time_steps must be non-empty and in [0,1], got {times}")
    if not eta_values or any(eta <= 0 for eta in eta_values):
        raise ValueError(f"stability.eta_values must be positive, got {eta_values}")
    if not step_values or any(steps <= 0 for steps in step_values):
        raise ValueError(
            f"stability.base_steps_values must be positive, got {step_values}"
        )
    if limit <= 0 or batch_size <= 0:
        raise ValueError("stability num_samples and batch_size must be positive")
    result = {
        "enabled": True,
        "num_samples": min(limit, len(z1)),
        "time_steps": times,
        "epsilon_sweep": {},
        "solver_sweep": {},
    }

    for eta in eta_values:
        key = f"{eta:.8g}"
        result["epsilon_sweep"][key] = {}
        for t_value in times:
            targets = evaluate_at_time(
                teacher, z0, z1, q, labels, t_value, batch_size, eta,
                main_steps, compute_dtype, device, limit,
            )
            result["epsilon_sweep"][key][f"{t_value:.8g}"] = summarize_targets(targets)

    eta_reference = min(eta_values, key=lambda value: abs(value - main_eta))
    eta_relative_differences = {}
    for eta in eta_values:
        eta_key = f"{eta:.8g}"
        eta_relative_differences[eta_key] = {}
        for t_value in times:
            t_key = f"{t_value:.8g}"
            reference = result["epsilon_sweep"][f"{eta_reference:.8g}"][t_key]["mean"]
            comparison = result["epsilon_sweep"][eta_key][t_key]["mean"]
            eta_relative_differences[eta_key][t_key] = abs(comparison - reference) / max(
                abs(reference), torch.finfo(torch.float64).eps
            )
    result["epsilon_relative_difference"] = {
        "reference_eta": eta_reference,
        "values": eta_relative_differences,
    }

    for steps in step_values:
        key = str(steps)
        result["solver_sweep"][key] = {}
        for t_value in times:
            targets = evaluate_at_time(
                teacher, z0, z1, q, labels, t_value, batch_size, main_eta,
                steps, compute_dtype, device, limit,
            )
            result["solver_sweep"][key][f"{t_value:.8g}"] = summarize_targets(targets)

    if len(step_values) >= 2:
        reference_steps = max(step_values)
        comparison_steps = min(step_values)
        relative_differences = {}
        for t_value in times:
            t_key = f"{t_value:.8g}"
            reference = result["solver_sweep"][str(reference_steps)][t_key]["mean"]
            comparison = result["solver_sweep"][str(comparison_steps)][t_key]["mean"]
            relative_differences[t_key] = abs(comparison - reference) / max(
                abs(reference), torch.finfo(torch.float64).eps
            )
        result["solver_relative_difference"] = {
            "comparison_base_steps": comparison_steps,
            "reference_base_steps": reference_steps,
            "values": relative_differences,
        }
        threshold = float(cfg.get("solver_relative_tolerance", 0.05))
        result["solver_relative_difference"]["tolerance"] = threshold
        result["solver_relative_difference"]["passed"] = all(
            value < threshold for value in relative_differences.values()
        )
    return result


def print_stability_summary(stability: dict) -> None:
    if not stability.get("enabled", False):
        print("Stability checks: disabled")
        return
    print("Finite-difference eta stability (relative to configured/reference eta):")
    epsilon_check = stability["epsilon_relative_difference"]
    for eta, values in epsilon_check["values"].items():
        formatted = ", ".join(f"t={t}: {value:.2%}" for t, value in values.items())
        print(f"  eta={eta}: {formatted}")
    solver_check = stability.get("solver_relative_difference")
    if solver_check is not None:
        formatted = ", ".join(
            f"t={t}: {value:.2%}" for t, value in solver_check["values"].items()
        )
        status = "PASS" if solver_check["passed"] else "WARNING: above tolerance"
        print(
            f"Solver stability {solver_check['comparison_base_steps']} vs "
            f"{solver_check['reference_base_steps']} steps: {formatted} [{status}]"
        )


def save_results(
    output_dir: str,
    times: Sequence[float],
    target_matrix: np.ndarray,
    summaries: Sequence[dict],
    sample_indices: np.ndarray,
    metadata: dict,
    stability: dict,
) -> None:
    means = np.asarray([row["mean"] for row in summaries], dtype=np.float64)
    stds = np.asarray([row["std"] for row in summaries], dtype=np.float64)
    standard_errors = np.asarray(
        [row["standard_error"] for row in summaries], dtype=np.float64
    )
    np.savez_compressed(
        os.path.join(output_dir, "a_t_targets.npz"),
        times=np.asarray(times, dtype=np.float64),
        targets=target_matrix,
        means=means,
        stds=stds,
        standard_errors=standard_errors,
        sample_indices=sample_indices,
    )

    rows = []
    for t_value, summary in zip(times, summaries):
        rows.append({"t": float(t_value), **summary})
    with open(os.path.join(output_dir, "a_t_summary.json"), "w") as f:
        json.dump({"metadata": metadata, "statistics": rows}, f, indent=2)
    with open(os.path.join(output_dir, "a_t_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(output_dir, "stability_checks.json"), "w") as f:
        json.dump(stability, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(times, means, marker="o", linewidth=2, label=r"$\widehat{a}(t)$")
    plt.fill_between(
        times,
        means - standard_errors,
        means + standard_errors,
        alpha=0.25,
        label="mean ± standard error",
    )
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1, label=r"$a(1)=1$")
    plt.xlabel("t")
    plt.ylabel(r"$\widehat{a}(t)$")
    plt.title("Frozen teacher flow-map sensitivity")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "a_t_curve.png"), dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate a(t) by central finite differences")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument("--device", default=None, help="Optional device override")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device or cfg.get("device", "cuda:0"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    output_dir = os.path.join(
        cfg["experiment"].get("output_dir", "./experiment"),
        cfg["experiment"]["name"],
    )
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy2(args.config, os.path.join(output_dir, "config.yaml"))

    ckpt_path = resolve_checkpoint_path(cfg["checkpoint"])
    print(f"Loading checkpoint metadata: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_args = resolve_model_args(checkpoint.get("args"), cfg["model_overrides"] or {})
    weight_key = "ema" if cfg["checkpoint"].get("use_ema", True) else "model"
    if weight_key not in checkpoint:
        raise KeyError(
            f"Checkpoint has no '{weight_key}' weights; available keys={list(checkpoint.keys())}"
        )
    teacher_weights = checkpoint[weight_key]
    del checkpoint
    print(
        f"Teacher: {model_args.model}, path={model_args.path_type}, "
        f"prediction={model_args.prediction}, image_size={model_args.image_size}"
    )

    eval_cfg = cfg["eval"]
    seed = int(eval_cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    z1, labels, sample_indices, data_source = collect_fixed_samples(
        cfg["data"], eval_cfg, model_args, device
    )
    expected_shape = (4, model_args.image_size // 8, model_args.image_size // 8)
    if tuple(z1.shape[1:]) != expected_shape:
        raise ValueError(
            f"Dataset latent shape is {tuple(z1.shape[1:])}; expected {expected_shape}"
        )
    if labels.min().item() < 0 or labels.max().item() >= model_args.num_classes:
        raise ValueError("Dataset labels fall outside the teacher's class range")
    z0, q = make_fixed_source_and_probes(z1, seed)
    print(f"Fixed samples: {len(z1)} from {data_source}; latent shape={tuple(z1.shape[1:])}")

    teacher = SiT_models[model_args.model](
        input_size=model_args.image_size // 8,
        num_classes=model_args.num_classes,
    ).to(device)
    teacher.load_state_dict(teacher_weights)
    del teacher_weights
    teacher.eval()
    teacher.requires_grad_(False)

    times = build_time_grid(eval_cfg)
    eta = float(eval_cfg.get("eta", 3e-3))
    base_steps = int(eval_cfg.get("base_steps", 32))
    batch_size = int(eval_cfg["batch_size"])
    if batch_size <= 0:
        raise ValueError("eval.batch_size must be positive")
    compute_dtype_name = str(eval_cfg.get("compute_dtype", "float32"))
    compute_dtype = parse_compute_dtype(compute_dtype_name)
    print(
        f"Evaluation: eta={eta:g}, base_steps={base_steps}, "
        f"compute_dtype={compute_dtype_name}, pair_batch={2 * batch_size}"
    )

    # Run the cheaper numerical checks before the formal N-sample sweep.
    stability = run_stability_checks(
        teacher,
        z0,
        z1,
        q,
        labels,
        cfg.get("stability", {}),
        eval_cfg,
        compute_dtype,
        device,
    )
    print_stability_summary(stability)

    target_rows, summaries = [], []
    for index, t_value in enumerate(times, start=1):
        targets = evaluate_at_time(
            teacher, z0, z1, q, labels, t_value, batch_size, eta,
            base_steps, compute_dtype, device,
        )
        summary = summarize_targets(targets)
        target_rows.append(targets.numpy())
        summaries.append(summary)
        print(
            f"[{index:02d}/{len(times):02d}] t={t_value:.5f}  "
            f"a_hat={summary['mean']:.8g}  se={summary['standard_error']:.3g}"
        )

    target_matrix = np.stack(target_rows, axis=0)
    if not np.isfinite(target_matrix).all():
        raise FloatingPointError("Non-finite finite-difference targets were produced")
    t1_index = next((i for i, t in enumerate(times) if t == 1.0), None)
    t1_mean = None if t1_index is None else summaries[t1_index]["mean"]
    if t1_mean is not None:
        print(f"Identity check: a(1)={t1_mean:.10f}, error={abs(t1_mean - 1.0):.3g}")

    metadata = {
        "checkpoint": os.path.abspath(ckpt_path),
        "weights": weight_key,
        "data_source": data_source,
        "model": vars(model_args),
        "eta": eta,
        "base_steps": base_steps,
        "solver": "heun",
        "compute_dtype": compute_dtype_name,
        "solver_state_dtype": "float32",
        "probe": "unit_norm_rademacher",
        "seed": seed,
        "a_t1_identity_error": None if t1_mean is None else abs(t1_mean - 1.0),
    }
    save_results(
        output_dir,
        times,
        target_matrix,
        summaries,
        sample_indices,
        metadata,
        stability,
    )
    print(f"Saved a(t) results to {output_dir}")


if __name__ == "__main__":
    main()
