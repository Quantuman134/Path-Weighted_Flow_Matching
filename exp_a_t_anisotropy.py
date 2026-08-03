"""Independent multi-probe experiment for frozen-teacher flow anisotropy.

For each fixed state z_t^(i), this experiment evaluates M normalized Gaussian
directions and stores

    A[t, i, m] = ||Phi(1, t; z_t^(i)) q[i, m]||_2^2.

It separates state-to-state variation from within-state directional variation
using the law of total variance, including the finite-M correction

    V_state = Var_i(mean_m A_im) - mean_i(Var_m A_im) / M.

The absolute directional fraction is complemented by the state-wise,
scale-normalized directional CV^2 = Var_m(A_im) / mean_m(A_im)^2.  This script
is deliberately separate from exp_a_t.py so the main a(t) estimator does not
pay the M-probe cost. Optionally, it also compares the same random-probe bank
with the normalized velocity residual between a frozen student and teacher.

Usage:
    python exp_a_t_anisotropy.py \
        --config configs/exp_a_t_anisotropy_config_XL_velocity_imagenet.yaml

    python exp_a_t_anisotropy.py \
        --config configs/exp_a_t_residual_alignment_config_XL_velocity_imagenet.yaml
"""

import argparse
import csv
import json
import math
import os
import shutil
from datetime import datetime, timezone
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from exp_a_t import (
    _autocast_context,
    collect_fixed_samples,
    evaluate_at_time,
    finite_difference_targets,
    format_duration,
    load_config,
    parse_compute_dtype,
    resolve_checkpoint_path,
    resolve_model_args,
    synchronize_device,
)
from models import SiT_models


def validate_eval_config(eval_cfg: dict) -> dict:
    """Parse and validate the anisotropy experiment parameters."""
    save_states_and_probes = eval_cfg.get("save_states_and_probes", True)
    if not isinstance(save_states_and_probes, bool):
        raise TypeError("eval.save_states_and_probes must be a YAML boolean")
    parsed = {
        "sample_pool_size": int(eval_cfg.get("sample_pool_size", 512)),
        "num_states": int(eval_cfg.get("num_states", 64)),
        "num_probes_per_state": int(eval_cfg.get("num_probes_per_state", 16)),
        "batch_size": int(eval_cfg.get("batch_size", 64)),
        "seed": int(eval_cfg.get("seed", 42)),
        "time_steps": [float(t) for t in eval_cfg["time_steps"]],
        "eta": float(eval_cfg.get("eta", 1e-3)),
        "base_steps": int(eval_cfg.get("base_steps", 128)),
        "compute_dtype": str(eval_cfg.get("compute_dtype", "float32")),
        "cv_epsilon": float(eval_cfg.get("cv_epsilon", 1e-12)),
        "save_states_and_probes": save_states_and_probes,
    }
    if parsed["sample_pool_size"] <= 0:
        raise ValueError("eval.sample_pool_size must be positive")
    if not 1 < parsed["num_states"] <= parsed["sample_pool_size"]:
        raise ValueError("eval.num_states must be in [2, sample_pool_size]")
    if parsed["num_probes_per_state"] < 2:
        raise ValueError("At least two probes per state are required to estimate variance")
    if parsed["batch_size"] <= 0:
        raise ValueError("eval.batch_size must be positive")
    if not parsed["time_steps"] or any(
        t < 0.0 or t > 1.0 for t in parsed["time_steps"]
    ):
        raise ValueError("eval.time_steps must be non-empty and lie in [0,1]")
    if len(set(parsed["time_steps"])) != len(parsed["time_steps"]):
        raise ValueError("eval.time_steps cannot contain duplicates")
    if parsed["time_steps"] != sorted(parsed["time_steps"]):
        raise ValueError("eval.time_steps must be strictly increasing")
    if parsed["eta"] <= 0.0 or parsed["base_steps"] <= 0:
        raise ValueError("eval.eta and eval.base_steps must be positive")
    if parsed["cv_epsilon"] <= 0.0:
        raise ValueError("eval.cv_epsilon must be positive")
    parse_compute_dtype(parsed["compute_dtype"])
    return parsed


def validate_residual_alignment_config(cfg: dict) -> dict:
    """Parse the optional residual-direction alignment pilot."""
    residual_cfg = cfg.get("residual_alignment", {})
    enabled = residual_cfg.get("enabled", False)
    save_directions = residual_cfg.get("save_residual_directions", True)
    if not isinstance(enabled, bool):
        raise TypeError("residual_alignment.enabled must be a YAML boolean")
    if not isinstance(save_directions, bool):
        raise TypeError(
            "residual_alignment.save_residual_directions must be a YAML boolean"
        )
    parsed = {
        "enabled": enabled,
        "min_residual_norm": float(
            residual_cfg.get("min_residual_norm", 1e-8)
        ),
        "save_residual_directions": save_directions,
    }
    if parsed["min_residual_norm"] <= 0.0:
        raise ValueError("residual_alignment.min_residual_norm must be positive")
    if enabled and "student_checkpoint" not in cfg:
        raise ValueError(
            "student_checkpoint is required when residual_alignment.enabled=true"
        )
    return parsed


def checkpoint_weight_key(checkpoint_cfg: dict) -> str:
    """Resolve an explicit checkpoint key or the legacy use_ema switch."""
    if checkpoint_cfg.get("weight_key") is not None:
        return str(checkpoint_cfg["weight_key"])
    return "ema" if checkpoint_cfg.get("use_ema", True) else "model"


def validate_student_model_args(teacher_args, student_args) -> None:
    """Require teacher/student fields needed for a pointwise velocity residual."""
    for field in ("image_size", "num_classes", "path_type", "prediction"):
        teacher_value = getattr(teacher_args, field)
        student_value = getattr(student_args, field)
        if student_value != teacher_value:
            raise ValueError(
                f"Student {field}={student_value!r} does not match "
                f"teacher {field}={teacher_value!r}"
            )


def make_fixed_states_and_probe_bank(
    z1_pool: torch.Tensor,
    num_states: int,
    seed: int,
    num_probes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return z0 [N,...] and unit Gaussian probes [N,M,...].

    z0 and the first probe are generated for the complete sample pool before
    slicing. Consequently they exactly match the first N rows of the one-probe
    experiment with the same pool size and seed. Additional probes are sampled
    independently for the selected N states.
    """
    source_generator = torch.Generator(device="cpu").manual_seed(seed + 2)
    probe_generator = torch.Generator(device="cpu").manual_seed(seed + 3)
    z0_pool = torch.randn(
        z1_pool.shape, generator=source_generator, dtype=torch.float32
    )
    first_probe_pool = torch.randn(
        z1_pool.shape, generator=probe_generator, dtype=torch.float32
    )
    z0 = z0_pool[:num_states].contiguous()
    first_probe = first_probe_pool[:num_states].contiguous()
    del z0_pool, first_probe_pool
    additional_probes = torch.randn(
        (num_probes - 1, num_states, *z1_pool.shape[1:]),
        generator=probe_generator,
        dtype=torch.float32,
    )
    probes_by_m = torch.cat((first_probe.unsqueeze(0), additional_probes), dim=0)
    norms = probes_by_m.flatten(2).norm(dim=2)
    if torch.any(norms == 0):
        raise RuntimeError("Sampled a zero-norm Gaussian probe")
    probes_by_m.div_(
        norms.reshape(num_probes, num_states, *([1] * (z1_pool.ndim - 1)))
    )
    probes = probes_by_m.transpose(0, 1).contiguous()
    return z0, probes


@torch.inference_mode()
def evaluate_multi_probe_at_time(
    teacher: torch.nn.Module,
    z0_cpu: torch.Tensor,
    z1_cpu: torch.Tensor,
    probes_cpu: torch.Tensor,
    labels_cpu: torch.Tensor,
    t_value: float,
    batch_size: int,
    eta: float,
    base_steps: int,
    compute_dtype,
    device: torch.device,
) -> torch.Tensor:
    """Evaluate A_im at one time, returning float64 CPU tensor [N,M]."""
    num_states, num_probes = probes_cpu.shape[:2]
    z_t = (1.0 - t_value) * z0_cpu + t_value * z1_cpu
    state_shape = z_t.shape[1:]
    z_t_pairs = z_t[:, None].expand(num_states, num_probes, *state_shape).reshape(
        num_states * num_probes, *state_shape
    )
    probe_pairs = probes_cpu.reshape(num_states * num_probes, *state_shape)
    label_pairs = labels_cpu[:, None].expand(num_states, num_probes).reshape(-1)

    targets = []
    for start in range(0, num_states * num_probes, batch_size):
        end = min(start + batch_size, num_states * num_probes)
        targets.append(
            finite_difference_targets(
                teacher=teacher,
                z_t=z_t_pairs[start:end].to(device, non_blocking=True),
                q=probe_pairs[start:end].to(device, non_blocking=True),
                t_start=t_value,
                labels=label_pairs[start:end].to(device, non_blocking=True),
                eta=eta,
                base_steps=base_steps,
                compute_dtype=compute_dtype,
            )
        )
    return torch.cat(targets).reshape(num_states, num_probes)


@torch.inference_mode()
def compute_residual_directions_at_time(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    z0_cpu: torch.Tensor,
    z1_cpu: torch.Tensor,
    labels_cpu: torch.Tensor,
    t_value: float,
    batch_size: int,
    compute_dtype,
    device: torch.device,
    min_residual_norm: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return unit student-minus-teacher residual directions for every state.

    States whose velocity residual norm is below ``min_residual_norm`` are
    marked invalid rather than normalizing numerical noise. Their direction is
    stored as zero and their aligned gain is omitted from downstream summaries.
    """
    directions = []
    residual_norm_squared = []
    valid_masks = []
    for start in range(0, len(z1_cpu), batch_size):
        end = min(start + batch_size, len(z1_cpu))
        z0 = z0_cpu[start:end].to(device, non_blocking=True)
        z1 = z1_cpu[start:end].to(device, non_blocking=True)
        labels = labels_cpu[start:end].to(device, non_blocking=True)
        z_t = ((1.0 - t_value) * z0 + t_value * z1).float()
        times = torch.full(
            (end - start,), t_value, device=device, dtype=torch.float32
        )
        with _autocast_context(device, compute_dtype):
            teacher_velocity = teacher(z_t, times, y=labels)
            student_velocity = student(z_t, times, y=labels)
        residual = student_velocity.float() - teacher_velocity.float()
        residual_norm = residual.flatten(1).norm(dim=1)
        valid = residual_norm >= min_residual_norm
        direction = torch.zeros_like(residual)
        if valid.any():
            norm_view = residual_norm[valid].reshape(
                int(valid.sum().item()), *([1] * (residual.ndim - 1))
            )
            direction[valid] = residual[valid] / norm_view
        directions.append(direction.cpu())
        residual_norm_squared.append(residual_norm.double().square().cpu())
        valid_masks.append(valid.cpu())
    return (
        torch.cat(directions),
        torch.cat(residual_norm_squared),
        torch.cat(valid_masks),
    )


@torch.inference_mode()
def evaluate_residual_alignment_at_time(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    z0_cpu: torch.Tensor,
    z1_cpu: torch.Tensor,
    labels_cpu: torch.Tensor,
    t_value: float,
    batch_size: int,
    eta: float,
    base_steps: int,
    compute_dtype,
    device: torch.device,
    min_residual_norm: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure teacher-flow gain along the normalized model residual direction."""
    directions, residual_norm_squared, valid = compute_residual_directions_at_time(
        teacher=teacher,
        student=student,
        z0_cpu=z0_cpu,
        z1_cpu=z1_cpu,
        labels_cpu=labels_cpu,
        t_value=t_value,
        batch_size=batch_size,
        compute_dtype=compute_dtype,
        device=device,
        min_residual_norm=min_residual_norm,
    )
    gains = torch.full((len(z1_cpu),), torch.nan, dtype=torch.float64)
    valid_indices = valid.nonzero(as_tuple=False).flatten()
    if valid_indices.numel() > 0:
        valid_gains = evaluate_at_time(
            teacher=teacher,
            z0_cpu=z0_cpu[valid_indices],
            z1_cpu=z1_cpu[valid_indices],
            q_cpu=directions[valid_indices],
            labels_cpu=labels_cpu[valid_indices],
            t_value=t_value,
            batch_size=batch_size,
            eta=eta,
            base_steps=base_steps,
            compute_dtype=compute_dtype,
            device=device,
        )
        gains[valid_indices] = valid_gains
    return gains, residual_norm_squared, valid, directions


def distribution_summary(values: torch.Tensor) -> dict:
    values = values.double().reshape(-1)
    count = int(values.numel())
    return {
        "count": count,
        "mean": values.mean().item(),
        "std": values.std(unbiased=True).item() if count > 1 else 0.0,
        "median": torch.quantile(values, 0.5).item(),
        "q10": torch.quantile(values, 0.1).item(),
        "q90": torch.quantile(values, 0.9).item(),
        "min": values.min().item(),
        "max": values.max().item(),
    }


def summarize_residual_alignment(
    random_targets: torch.Tensor,
    residual_gains: torch.Tensor,
    residual_norm_squared: torch.Tensor,
    residual_valid: torch.Tensor,
    denominator_epsilon: float,
) -> Tuple[dict, Dict[str, torch.Tensor]]:
    """Compare residual-direction gain against the same state's random probes."""
    random_values = random_targets.double()
    random_mean = random_values.mean(dim=1)
    valid = (
        residual_valid.bool()
        & torch.isfinite(residual_gains)
        & torch.isfinite(residual_norm_squared)
        & (random_mean > denominator_epsilon)
    )
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        raise RuntimeError("No valid residual directions remain after filtering")

    alignment_ratio = torch.full_like(residual_gains, torch.nan)
    probe_percentile = torch.full_like(residual_gains, torch.nan)
    propagated_residual_energy = torch.full_like(residual_gains, torch.nan)
    random_baseline_energy = torch.full_like(residual_gains, torch.nan)
    alignment_ratio[valid] = residual_gains[valid] / random_mean[valid]
    probe_percentile[valid] = (
        random_values[valid] <= residual_gains[valid, None]
    ).double().mean(dim=1)
    propagated_residual_energy[valid] = (
        residual_norm_squared[valid] * residual_gains[valid]
    )
    random_baseline_energy[valid] = (
        residual_norm_squared[valid] * random_mean[valid]
    )

    valid_random_mean = random_mean[valid]
    valid_residual_gain = residual_gains[valid]
    valid_norm_squared = residual_norm_squared[valid]
    valid_alignment_ratio = alignment_ratio[valid]
    valid_percentile = probe_percentile[valid]
    valid_propagated_energy = propagated_residual_energy[valid]
    valid_baseline_energy = random_baseline_energy[valid]
    summary = {
        "num_states": int(len(random_mean)),
        "num_valid_states": valid_count,
        "num_filtered_states": int(len(random_mean) - valid_count),
        "mean_random_gain": valid_random_mean.mean().item(),
        "mean_residual_aligned_gain": valid_residual_gain.mean().item(),
        "ratio_of_mean_gains": (
            valid_residual_gain.mean() / valid_random_mean.mean()
        ).item(),
        "magnitude_weighted_alignment_ratio": (
            valid_propagated_energy.sum()
            / valid_baseline_energy.sum().clamp_min(denominator_epsilon)
        ).item(),
        "fraction_alignment_ratio_above_one": (
            valid_alignment_ratio > 1.0
        ).double().mean().item(),
        "fraction_probe_percentile_at_least_0_9": (
            valid_percentile >= 0.9
        ).double().mean().item(),
        "residual_norm_squared_statistics": distribution_summary(valid_norm_squared),
        "random_gain_statistics": distribution_summary(valid_random_mean),
        "residual_aligned_gain_statistics": distribution_summary(valid_residual_gain),
        "alignment_ratio_statistics": distribution_summary(valid_alignment_ratio),
        "probe_percentile_statistics": distribution_summary(valid_percentile),
        "propagated_residual_energy_statistics": distribution_summary(
            valid_propagated_energy
        ),
    }
    per_state = {
        "valid": valid,
        "residual_norm_squared": residual_norm_squared.double(),
        "random_gain": random_mean,
        "residual_aligned_gain": residual_gains.double(),
        "alignment_ratio": alignment_ratio,
        "probe_percentile": probe_percentile,
        "propagated_residual_energy": propagated_residual_energy,
        "random_baseline_energy": random_baseline_energy,
    }
    return summary, per_state


def summarize_anisotropy(
    targets: torch.Tensor,
    cv_epsilon: float,
) -> Tuple[dict, Dict[str, torch.Tensor]]:
    """Compute corrected state/direction variance components for [N,M]."""
    values = targets.double()
    num_states, num_probes = values.shape
    state_means = values.mean(dim=1)
    state_variances = values.var(dim=1, unbiased=True)
    state_cv2 = state_variances / (state_means.square() + cv_epsilon)
    state_cv = state_cv2.sqrt()

    directional_variance = state_variances.mean().item()
    observed_state_mean_variance = state_means.var(unbiased=True).item()
    finite_probe_correction = directional_variance / num_probes
    uncorrected_state_variance = (
        observed_state_mean_variance - finite_probe_correction
    )
    state_variance = max(0.0, uncorrected_state_variance)
    total_variance = state_variance + directional_variance
    if total_variance > cv_epsilon:
        state_fraction = state_variance / total_variance
        directional_fraction = directional_variance / total_variance
    else:
        state_fraction = None
        directional_fraction = None

    estimator_variance = (
        state_variance / num_states
        + directional_variance / (num_states * num_probes)
    )
    summary = {
        "num_states": num_states,
        "num_probes_per_state": num_probes,
        "mean_a": state_means.mean().item(),
        "hierarchical_standard_error": math.sqrt(max(0.0, estimator_variance)),
        "variance_components": {
            "state": state_variance,
            "directional": directional_variance,
            "total": total_variance,
            "state_fraction": state_fraction,
            "directional_fraction": directional_fraction,
            "observed_state_mean_variance": observed_state_mean_variance,
            "finite_probe_correction": finite_probe_correction,
            "uncorrected_state_variance": uncorrected_state_variance,
            "state_variance_clamped_to_zero": uncorrected_state_variance < 0.0,
        },
        "raw_pair_statistics": distribution_summary(values),
        "state_mean_statistics": distribution_summary(state_means),
        "state_directional_variance_statistics": distribution_summary(
            state_variances
        ),
        "state_directional_cv2_statistics": distribution_summary(state_cv2),
        "state_directional_cv_statistics": distribution_summary(state_cv),
    }
    per_state = {
        "means": state_means,
        "variances": state_variances,
        "cv2": state_cv2,
        "cv": state_cv,
        "minimum": values.min(dim=1).values,
        "maximum": values.max(dim=1).values,
    }
    return summary, per_state


def save_outputs(
    output_dir: str,
    times: Sequence[float],
    targets: torch.Tensor,
    per_state_by_time: List[Dict[str, torch.Tensor]],
    summaries: List[dict],
    sample_indices: np.ndarray,
    labels: torch.Tensor,
    z0: torch.Tensor,
    z1: torch.Tensor,
    probes: torch.Tensor,
    metadata: dict,
    save_states_and_probes: bool,
) -> None:
    state_means = torch.stack([row["means"] for row in per_state_by_time])
    state_variances = torch.stack([row["variances"] for row in per_state_by_time])
    state_cv2 = torch.stack([row["cv2"] for row in per_state_by_time])
    state_cv = torch.stack([row["cv"] for row in per_state_by_time])
    elapsed = np.asarray([row["elapsed_seconds"] for row in summaries], dtype=np.float64)
    npz_values = {
        "times": np.asarray(times, dtype=np.float64),
        "targets": targets.numpy(),
        "state_means": state_means.numpy(),
        "state_variances": state_variances.numpy(),
        "state_cv2": state_cv2.numpy(),
        "state_cv": state_cv.numpy(),
        "sample_indices": sample_indices,
        "labels": labels.numpy(),
        "timepoint_elapsed_seconds": elapsed,
    }
    if save_states_and_probes:
        npz_values.update({
            "z0": z0.numpy(),
            "z1": z1.numpy(),
            "probes": probes.numpy(),
        })
    np.savez_compressed(
        os.path.join(output_dir, "anisotropy_targets.npz"), **npz_values
    )

    with open(os.path.join(output_dir, "anisotropy_summary.json"), "w") as f:
        json.dump({"metadata": metadata, "statistics": summaries}, f, indent=2)

    summary_rows = []
    for t_value, summary in zip(times, summaries):
        components = summary["variance_components"]
        cv2_stats = summary["state_directional_cv2_statistics"]
        summary_rows.append({
            "t": t_value,
            "mean_a": summary["mean_a"],
            "hierarchical_standard_error": summary["hierarchical_standard_error"],
            "state_variance": components["state"],
            "directional_variance": components["directional"],
            "state_fraction": components["state_fraction"],
            "directional_fraction": components["directional_fraction"],
            "median_state_cv2": cv2_stats["median"],
            "q10_state_cv2": cv2_stats["q10"],
            "q90_state_cv2": cv2_stats["q90"],
            "elapsed_seconds": summary["elapsed_seconds"],
        })
    with open(os.path.join(output_dir, "anisotropy_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    state_rows = []
    for time_index, t_value in enumerate(times):
        state_data = per_state_by_time[time_index]
        for state_index in range(len(sample_indices)):
            state_rows.append({
                "t": t_value,
                "state_index": state_index,
                "dataset_index": int(sample_indices[state_index]),
                "label": int(labels[state_index]),
                "directional_mean": state_data["means"][state_index].item(),
                "directional_variance": state_data["variances"][state_index].item(),
                "directional_cv2": state_data["cv2"][state_index].item(),
                "directional_cv": state_data["cv"][state_index].item(),
                "minimum": state_data["minimum"][state_index].item(),
                "maximum": state_data["maximum"][state_index].item(),
            })
    with open(
        os.path.join(output_dir, "anisotropy_state_statistics.csv"), "w", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(state_rows[0].keys()))
        writer.writeheader()
        writer.writerows(state_rows)

    plot_diagnostics(output_dir, times, summaries)


def save_residual_alignment_outputs(
    output_dir: str,
    times: Sequence[float],
    residual_gains: torch.Tensor,
    residual_norm_squared: torch.Tensor,
    residual_directions: torch.Tensor,
    per_state_by_time: List[Dict[str, torch.Tensor]],
    summaries: List[dict],
    sample_indices: np.ndarray,
    labels: torch.Tensor,
    metadata: dict,
    save_residual_directions: bool,
) -> None:
    """Save the residual-alignment pilot separately from anisotropy outputs."""
    random_gain = torch.stack([row["random_gain"] for row in per_state_by_time])
    alignment_ratio = torch.stack(
        [row["alignment_ratio"] for row in per_state_by_time]
    )
    probe_percentile = torch.stack(
        [row["probe_percentile"] for row in per_state_by_time]
    )
    propagated_energy = torch.stack(
        [row["propagated_residual_energy"] for row in per_state_by_time]
    )
    random_baseline_energy = torch.stack(
        [row["random_baseline_energy"] for row in per_state_by_time]
    )
    valid = torch.stack([row["valid"] for row in per_state_by_time])
    npz_values = {
        "times": np.asarray(times, dtype=np.float64),
        "residual_norm_squared": residual_norm_squared.numpy(),
        "residual_aligned_gain": residual_gains.numpy(),
        "random_gain": random_gain.numpy(),
        "alignment_ratio": alignment_ratio.numpy(),
        "probe_percentile": probe_percentile.numpy(),
        "propagated_residual_energy": propagated_energy.numpy(),
        "random_baseline_energy": random_baseline_energy.numpy(),
        "valid": valid.numpy(),
        "sample_indices": sample_indices,
        "labels": labels.numpy(),
    }
    if save_residual_directions:
        npz_values["residual_directions"] = residual_directions.numpy()
    np.savez_compressed(
        os.path.join(output_dir, "residual_alignment_targets.npz"), **npz_values
    )

    with open(
        os.path.join(output_dir, "residual_alignment_summary.json"), "w"
    ) as f:
        json.dump({"metadata": metadata, "statistics": summaries}, f, indent=2)

    summary_rows = []
    for t_value, summary in zip(times, summaries):
        gamma_stats = summary["alignment_ratio_statistics"]
        percentile_stats = summary["probe_percentile_statistics"]
        norm_stats = summary["residual_norm_squared_statistics"]
        summary_rows.append({
            "t": t_value,
            "num_states": summary["num_states"],
            "num_valid_states": summary["num_valid_states"],
            "num_filtered_states": summary["num_filtered_states"],
            "mean_random_gain": summary["mean_random_gain"],
            "mean_residual_aligned_gain": summary["mean_residual_aligned_gain"],
            "ratio_of_mean_gains": summary["ratio_of_mean_gains"],
            "magnitude_weighted_alignment_ratio": summary[
                "magnitude_weighted_alignment_ratio"
            ],
            "median_alignment_ratio": gamma_stats["median"],
            "q10_alignment_ratio": gamma_stats["q10"],
            "q90_alignment_ratio": gamma_stats["q90"],
            "fraction_alignment_ratio_above_one": summary[
                "fraction_alignment_ratio_above_one"
            ],
            "median_probe_percentile": percentile_stats["median"],
            "q10_probe_percentile": percentile_stats["q10"],
            "q90_probe_percentile": percentile_stats["q90"],
            "fraction_probe_percentile_at_least_0_9": summary[
                "fraction_probe_percentile_at_least_0_9"
            ],
            "median_residual_norm_squared": norm_stats["median"],
            "elapsed_seconds": summary["elapsed_seconds"],
        })
    with open(
        os.path.join(output_dir, "residual_alignment_summary.csv"), "w", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    state_rows = []
    for time_index, t_value in enumerate(times):
        state_data = per_state_by_time[time_index]
        for state_index in range(len(sample_indices)):
            state_rows.append({
                "t": t_value,
                "state_index": state_index,
                "dataset_index": int(sample_indices[state_index]),
                "label": int(labels[state_index]),
                "valid": bool(state_data["valid"][state_index]),
                "residual_norm_squared": state_data[
                    "residual_norm_squared"
                ][state_index].item(),
                "residual_aligned_gain": state_data[
                    "residual_aligned_gain"
                ][state_index].item(),
                "random_gain": state_data["random_gain"][state_index].item(),
                "alignment_ratio": state_data[
                    "alignment_ratio"
                ][state_index].item(),
                "probe_percentile": state_data[
                    "probe_percentile"
                ][state_index].item(),
                "propagated_residual_energy": state_data[
                    "propagated_residual_energy"
                ][state_index].item(),
                "random_baseline_energy": state_data[
                    "random_baseline_energy"
                ][state_index].item(),
            })
    with open(
        os.path.join(output_dir, "residual_alignment_state_statistics.csv"),
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(state_rows[0].keys()))
        writer.writeheader()
        writer.writerows(state_rows)

    plot_residual_alignment(output_dir, times, summaries)


def plot_residual_alignment(
    output_dir: str,
    times: Sequence[float],
    summaries: List[dict],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    random_gain = [row["mean_random_gain"] for row in summaries]
    residual_gain = [row["mean_residual_aligned_gain"] for row in summaries]
    gamma_median = [row["alignment_ratio_statistics"]["median"] for row in summaries]
    gamma_q10 = [row["alignment_ratio_statistics"]["q10"] for row in summaries]
    gamma_q90 = [row["alignment_ratio_statistics"]["q90"] for row in summaries]
    percentile_median = [
        row["probe_percentile_statistics"]["median"] for row in summaries
    ]
    percentile_q10 = [row["probe_percentile_statistics"]["q10"] for row in summaries]
    percentile_q90 = [row["probe_percentile_statistics"]["q90"] for row in summaries]

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(times, random_gain, marker="o", label="mean random gain")
    axes[0].plot(times, residual_gain, marker="o", label="mean residual gain")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("flow-map gain")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(times, gamma_median, marker="o", label=r"median $\gamma_i$")
    axes[1].fill_between(times, gamma_q10, gamma_q90, alpha=0.25, label="q10–q90")
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("residual/random gain ratio")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        times, percentile_median, marker="o", label="median probe percentile"
    )
    axes[2].fill_between(
        times, percentile_q10, percentile_q90, alpha=0.25, label="q10–q90"
    )
    axes[2].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("residual direction percentile")
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    figure.suptitle("Residual alignment with frozen-teacher flow-map gain")
    figure.tight_layout()
    figure.savefig(
        os.path.join(output_dir, "residual_alignment_diagnostics.png"), dpi=180
    )
    plt.close(figure)


def plot_diagnostics(output_dir: str, times: Sequence[float], summaries: List[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state_fraction = [row["variance_components"]["state_fraction"] for row in summaries]
    direction_fraction = [
        row["variance_components"]["directional_fraction"] for row in summaries
    ]
    state_fraction = [np.nan if value is None else value for value in state_fraction]
    direction_fraction = [np.nan if value is None else value for value in direction_fraction]
    cv2_median = [row["state_directional_cv2_statistics"]["median"] for row in summaries]
    cv2_q10 = [row["state_directional_cv2_statistics"]["q10"] for row in summaries]
    cv2_q90 = [row["state_directional_cv2_statistics"]["q90"] for row in summaries]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(times, state_fraction, marker="o", label="state fraction")
    axes[0].plot(times, direction_fraction, marker="o", label="direction fraction")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("fraction of total A variation")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(times, cv2_median, marker="o", label=r"median state $CV_{dir}^2$")
    axes[1].fill_between(times, cv2_q10, cv2_q90, alpha=0.25, label="state q10–q90")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel(r"state-wise $CV_{dir}^2$")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.suptitle("Frozen teacher flow-map anisotropy diagnostics")
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "anisotropy_diagnostics.png"), dpi=180)
    plt.close(figure)


def main() -> None:
    experiment_started_at = datetime.now(timezone.utc)
    experiment_start = perf_counter()
    parser = argparse.ArgumentParser(description="Multi-probe flow-map anisotropy")
    parser.add_argument("--config", required=True, help="Path to anisotropy YAML")
    parser.add_argument("--device", default=None, help="Optional device override")
    cli = parser.parse_args()

    cfg = load_config(cli.config)
    eval_cfg = validate_eval_config(cfg["eval"])
    residual_cfg = validate_residual_alignment_config(cfg)
    device = torch.device(cli.device or cfg.get("device", "cuda:0"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    compute_dtype = parse_compute_dtype(eval_cfg["compute_dtype"])

    output_dir = os.path.join(
        cfg["experiment"].get("output_dir", "./experiment"),
        cfg["experiment"]["name"],
    )
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy2(cli.config, os.path.join(output_dir, "config.yaml"))

    ckpt_path = resolve_checkpoint_path(cfg["checkpoint"])
    print(f"Loading checkpoint metadata: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_args = resolve_model_args(checkpoint.get("args"), cfg["model_overrides"] or {})
    weight_key = checkpoint_weight_key(cfg["checkpoint"])
    if weight_key not in checkpoint:
        raise KeyError(
            f"Checkpoint has no '{weight_key}' weights; available keys={list(checkpoint.keys())}"
        )
    teacher_weights = checkpoint[weight_key]

    student_ckpt_path = None
    student_weight_key = None
    student_model_args = None
    student_weights = None
    if residual_cfg["enabled"]:
        student_ckpt_path = resolve_checkpoint_path(cfg["student_checkpoint"])
        if os.path.abspath(student_ckpt_path) == os.path.abspath(ckpt_path):
            student_checkpoint = checkpoint
        else:
            student_checkpoint = torch.load(
                student_ckpt_path, map_location="cpu", weights_only=False
            )
        student_model_args = resolve_model_args(
            student_checkpoint.get("args"),
            cfg.get("student_model_overrides") or {},
        )
        validate_student_model_args(model_args, student_model_args)
        student_weight_key = checkpoint_weight_key(cfg["student_checkpoint"])
        if student_weight_key not in student_checkpoint:
            raise KeyError(
                f"Student checkpoint has no '{student_weight_key}' weights; "
                f"available keys={list(student_checkpoint.keys())}"
            )
        student_weights = student_checkpoint[student_weight_key]
        if student_checkpoint is not checkpoint:
            del student_checkpoint
    del checkpoint

    seed = eval_cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    pool_eval_cfg = {
        "num_samples": eval_cfg["sample_pool_size"],
        "batch_size": eval_cfg["batch_size"],
        "seed": seed,
    }
    z1_pool, labels_pool, sample_indices_pool, data_source = collect_fixed_samples(
        cfg["data"], pool_eval_cfg, model_args, device
    )
    num_states = eval_cfg["num_states"]
    z0, probes = make_fixed_states_and_probe_bank(
        z1_pool, num_states, seed, eval_cfg["num_probes_per_state"]
    )
    z1 = z1_pool[:num_states].contiguous()
    labels = labels_pool[:num_states].contiguous()
    sample_indices = sample_indices_pool[:num_states].copy()
    del z1_pool, labels_pool, sample_indices_pool
    expected_shape = (4, model_args.image_size // 8, model_args.image_size // 8)
    if tuple(z1.shape[1:]) != expected_shape:
        raise ValueError(
            f"Dataset latent shape is {tuple(z1.shape[1:])}; expected {expected_shape}"
        )
    if labels.min().item() < 0 or labels.max().item() >= model_args.num_classes:
        raise ValueError("Dataset labels fall outside the teacher's class range")
    probe_norm_error = (
        probes.flatten(2).norm(dim=2).sub(1.0).abs().max().item()
    )

    teacher = SiT_models[model_args.model](
        input_size=model_args.image_size // 8,
        num_classes=model_args.num_classes,
    ).to(device)
    teacher.load_state_dict(teacher_weights)
    del teacher_weights
    teacher.eval()
    teacher.requires_grad_(False)
    student = None
    if residual_cfg["enabled"]:
        student = SiT_models[student_model_args.model](
            input_size=student_model_args.image_size // 8,
            num_classes=student_model_args.num_classes,
        ).to(device)
        student.load_state_dict(student_weights)
        del student_weights
        student.eval()
        student.requires_grad_(False)
    synchronize_device(device)
    initialization_seconds = perf_counter() - experiment_start

    print(
        f"Anisotropy experiment: states={num_states}, "
        f"probes/state={eval_cfg['num_probes_per_state']}, "
        f"pair_batch={eval_cfg['batch_size']}, "
        f"teacher_batch={2 * eval_cfg['batch_size']}, "
        f"eta={eval_cfg['eta']:g}, base_steps={eval_cfg['base_steps']}, "
        f"dtype={eval_cfg['compute_dtype']}"
    )
    if residual_cfg["enabled"]:
        print(
            "Residual alignment enabled: "
            f"student={student_model_args.model}, "
            f"weights={student_weight_key}, checkpoint={student_ckpt_path}, "
            f"min_norm={residual_cfg['min_residual_norm']:g}"
        )
    print(
        f"State pool: first {num_states} of {eval_cfg['sample_pool_size']} fixed samples "
        f"from {data_source}"
    )

    targets_by_time, summaries, per_state_by_time = [], [], []
    residual_gains_by_time = []
    residual_norm_squared_by_time = []
    residual_directions_by_time = []
    residual_summaries = []
    residual_per_state_by_time = []
    evaluation_start = perf_counter()
    for index, t_value in enumerate(eval_cfg["time_steps"], start=1):
        synchronize_device(device)
        timepoint_start = perf_counter()
        random_probe_start = perf_counter()
        targets = evaluate_multi_probe_at_time(
            teacher,
            z0,
            z1,
            probes,
            labels,
            t_value,
            eval_cfg["batch_size"],
            eval_cfg["eta"],
            eval_cfg["base_steps"],
            compute_dtype,
            device,
        )
        synchronize_device(device)
        random_probe_seconds = perf_counter() - random_probe_start

        residual_summary = None
        if residual_cfg["enabled"]:
            residual_start = perf_counter()
            (
                residual_gains,
                residual_norm_squared,
                residual_valid,
                residual_directions,
            ) = evaluate_residual_alignment_at_time(
                teacher=teacher,
                student=student,
                z0_cpu=z0,
                z1_cpu=z1,
                labels_cpu=labels,
                t_value=t_value,
                batch_size=eval_cfg["batch_size"],
                eta=eval_cfg["eta"],
                base_steps=eval_cfg["base_steps"],
                compute_dtype=compute_dtype,
                device=device,
                min_residual_norm=residual_cfg["min_residual_norm"],
            )
            synchronize_device(device)
            residual_seconds = perf_counter() - residual_start
            residual_summary, residual_per_state = summarize_residual_alignment(
                random_targets=targets,
                residual_gains=residual_gains,
                residual_norm_squared=residual_norm_squared,
                residual_valid=residual_valid,
                denominator_epsilon=eval_cfg["cv_epsilon"],
            )
            valid_directions = residual_directions[residual_valid]
            direction_norm_error = (
                valid_directions.flatten(1).norm(dim=1)
                .sub(1.0).abs().max().item()
            )
            residual_summary["t"] = t_value
            residual_summary["elapsed_seconds"] = residual_seconds
            residual_summary["maximum_unit_direction_norm_error"] = (
                direction_norm_error
            )
            residual_gains_by_time.append(residual_gains)
            residual_norm_squared_by_time.append(residual_norm_squared)
            residual_directions_by_time.append(residual_directions)
            residual_summaries.append(residual_summary)
            residual_per_state_by_time.append(residual_per_state)

        elapsed_seconds = perf_counter() - timepoint_start
        summary, per_state = summarize_anisotropy(targets, eval_cfg["cv_epsilon"])
        summary["t"] = t_value
        summary["elapsed_seconds"] = elapsed_seconds
        summary["random_probe_elapsed_seconds"] = random_probe_seconds
        targets_by_time.append(targets)
        summaries.append(summary)
        per_state_by_time.append(per_state)
        components = summary["variance_components"]
        print(
            f"[{index:02d}/{len(eval_cfg['time_steps']):02d}] t={t_value:.5f} "
            f"mean={summary['mean_a']:.6g} "
            f"R_state={components['state_fraction']} "
            f"R_direction={components['directional_fraction']} "
            f"median_CV2={summary['state_directional_cv2_statistics']['median']:.4g} "
            f"elapsed={format_duration(elapsed_seconds)}"
        )
        if residual_summary is not None:
            print(
                "    residual alignment: "
                f"valid={residual_summary['num_valid_states']}/{num_states} "
                f"median_gamma={residual_summary['alignment_ratio_statistics']['median']:.4g} "
                f"median_percentile={residual_summary['probe_percentile_statistics']['median']:.4g} "
                f"magnitude_weighted_gamma="
                f"{residual_summary['magnitude_weighted_alignment_ratio']:.4g}"
            )
    synchronize_device(device)
    evaluation_seconds = perf_counter() - evaluation_start
    target_tensor = torch.stack(targets_by_time)
    if not torch.isfinite(target_tensor).all():
        raise FloatingPointError("Non-finite anisotropy targets were produced")

    residual_gain_tensor = None
    residual_norm_squared_tensor = None
    residual_direction_tensor = None
    if residual_cfg["enabled"]:
        residual_gain_tensor = torch.stack(residual_gains_by_time)
        residual_norm_squared_tensor = torch.stack(residual_norm_squared_by_time)
        residual_direction_tensor = torch.stack(residual_directions_by_time)
        residual_valid_tensor = torch.stack(
            [row["valid"] for row in residual_per_state_by_time]
        )
        if not torch.isfinite(residual_gain_tensor[residual_valid_tensor]).all():
            raise FloatingPointError(
                "Non-finite residual-aligned gains were produced for valid states"
            )
        if not torch.isfinite(residual_norm_squared_tensor).all():
            raise FloatingPointError("Non-finite residual norms were produced")

    t1_index = next(
        (i for i, t in enumerate(eval_cfg["time_steps"]) if t == 1.0), None
    )
    identity_error = (
        None if t1_index is None
        else abs(summaries[t1_index]["mean_a"] - 1.0)
    )
    metadata = {
        "checkpoint": os.path.abspath(ckpt_path),
        "weights": weight_key,
        "data_source": data_source,
        "model": vars(model_args),
        "sample_pool_size": eval_cfg["sample_pool_size"],
        "state_selection": "first_num_states_from_fixed_pool",
        "num_states": num_states,
        "num_probes_per_state": eval_cfg["num_probes_per_state"],
        "probe": "unit_norm_gaussian",
        "probe_first_column_matches_one_probe_experiment": True,
        "maximum_probe_norm_error": probe_norm_error,
        "eta": eval_cfg["eta"],
        "base_steps": eval_cfg["base_steps"],
        "solver": "heun",
        "compute_dtype": eval_cfg["compute_dtype"],
        "solver_state_dtype": "float32",
        "seed": seed,
        "a_t1_identity_error": identity_error,
        "variance_fraction_note": (
            "directional_fraction measures the directional contribution to absolute "
            "A variation; state_directional_cv2 is the scale-normalized diagnostic"
        ),
        "residual_alignment_enabled": residual_cfg["enabled"],
        "timing_file": "anisotropy_timing.json",
    }

    save_start = perf_counter()
    save_outputs(
        output_dir,
        eval_cfg["time_steps"],
        target_tensor,
        per_state_by_time,
        summaries,
        sample_indices,
        labels,
        z0,
        z1,
        probes,
        metadata,
        eval_cfg["save_states_and_probes"],
    )
    if residual_cfg["enabled"]:
        residual_metadata = {
            "teacher_checkpoint": os.path.abspath(ckpt_path),
            "teacher_weights": weight_key,
            "teacher_model": vars(model_args),
            "student_checkpoint": os.path.abspath(student_ckpt_path),
            "student_weights": student_weight_key,
            "student_model": vars(student_model_args),
            "residual_definition": "student_velocity_minus_teacher_velocity",
            "residual_direction": "unit_norm_for_nonfiltered_states",
            "min_residual_norm": residual_cfg["min_residual_norm"],
            "random_baseline": (
                "per-state mean over the same normalized Gaussian probe bank"
            ),
            "num_states": num_states,
            "num_random_probes_per_state": eval_cfg["num_probes_per_state"],
            "sample_pool_size": eval_cfg["sample_pool_size"],
            "state_selection": "first_num_states_from_fixed_pool",
            "time_steps": eval_cfg["time_steps"],
            "eta": eval_cfg["eta"],
            "base_steps": eval_cfg["base_steps"],
            "solver": "heun",
            "compute_dtype": eval_cfg["compute_dtype"],
            "solver_state_dtype": "float32",
            "seed": seed,
            "probe_percentile_definition": (
                "mean_m[ random_gain_im <= residual_aligned_gain_i ]"
            ),
            "timing_file": "anisotropy_timing.json",
        }
        save_residual_alignment_outputs(
            output_dir=output_dir,
            times=eval_cfg["time_steps"],
            residual_gains=residual_gain_tensor,
            residual_norm_squared=residual_norm_squared_tensor,
            residual_directions=residual_direction_tensor,
            per_state_by_time=residual_per_state_by_time,
            summaries=residual_summaries,
            sample_indices=sample_indices,
            labels=labels,
            metadata=residual_metadata,
            save_residual_directions=residual_cfg[
                "save_residual_directions"
            ],
        )
    save_seconds = perf_counter() - save_start
    finished_at = datetime.now(timezone.utc)
    total_seconds = perf_counter() - experiment_start
    timing = {
        "started_at_utc": experiment_started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "initialization_seconds": initialization_seconds,
        "evaluation_seconds": evaluation_seconds,
        "timepoint_seconds": {
            f"{t:.8g}": row["elapsed_seconds"]
            for t, row in zip(eval_cfg["time_steps"], summaries)
        },
        "random_probe_timepoint_seconds": {
            f"{t:.8g}": row["random_probe_elapsed_seconds"]
            for t, row in zip(eval_cfg["time_steps"], summaries)
        },
        "residual_alignment_timepoint_seconds": (
            {
                f"{t:.8g}": row["elapsed_seconds"]
                for t, row in zip(eval_cfg["time_steps"], residual_summaries)
            }
            if residual_cfg["enabled"] else {}
        ),
        "results_save_seconds": save_seconds,
        "total_wall_seconds": total_seconds,
        "allocated_gpu_count": 1,
        "single_gpu_wall_hours": total_seconds / 3600.0,
    }
    with open(os.path.join(output_dir, "anisotropy_timing.json"), "w") as f:
        json.dump(timing, f, indent=2)
    print(f"Saved anisotropy results to {output_dir}")
    if residual_cfg["enabled"]:
        print(f"Saved residual-alignment results to {output_dir}")
    print(
        f"Total elapsed: {format_duration(total_seconds)} "
        f"({total_seconds / 3600.0:.4f} single-GPU hours)"
    )


if __name__ == "__main__":
    main()
