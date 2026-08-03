"""Check whether saved residual directions align with teacher velocity.

This is a cheap post-processing pass for an existing residual-alignment run.
It reloads the saved states and unit residual directions, evaluates the frozen
teacher velocity once at each state, and reports

    cos^2(rho, v_teacher),
    rho_parallel / rho_perpendicular norms,
    c = <rho, v_teacher> / ||v_teacher||^2.

No ODE rollout, finite difference, student model, JVP, or backward pass is
used. The teacher checkpoint and weight key are read from the existing
``residual_alignment_summary.json`` unless overridden on the command line.

Usage:
    python analyze_residual_velocity_alignment.py \
        --results-dir experiment/a_t_residual_alignment_SiT_XL_velocity_imagenet
"""

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from time import perf_counter
from typing import Dict, Optional

import numpy as np
import torch

from exp_a_t import (
    _autocast_context,
    format_duration,
    parse_compute_dtype,
    resolve_checkpoint_path,
    resolve_model_args,
    synchronize_device,
)
from models import SiT_models


def distribution_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot summarize an empty finite array")
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.quantile(values, 0.5)),
        "q10": float(np.quantile(values, 0.1)),
        "q90": float(np.quantile(values, 0.9)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2 or x.std() == 0.0 or y.std() == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, including deterministic tie handling."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return None
    return pearson_correlation(average_ranks(x[valid]), average_ranks(y[valid]))


def load_existing_results(results_dir: str) -> Dict[str, np.ndarray]:
    anisotropy_path = os.path.join(results_dir, "anisotropy_targets.npz")
    residual_path = os.path.join(results_dir, "residual_alignment_targets.npz")
    summary_path = os.path.join(results_dir, "residual_alignment_summary.json")
    for path in (anisotropy_path, residual_path, summary_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required existing result is missing: {path}")

    with np.load(anisotropy_path, allow_pickle=False) as source:
        required = ("times", "z0", "z1", "labels", "sample_indices")
        missing = [key for key in required if key not in source]
        if missing:
            raise KeyError(f"{anisotropy_path} is missing arrays: {missing}")
        arrays = {key: source[key].copy() for key in required}
    with np.load(residual_path, allow_pickle=False) as source:
        required = (
            "times",
            "labels",
            "sample_indices",
            "residual_directions",
            "residual_norm_squared",
            "alignment_ratio",
            "residual_aligned_gain",
            "propagated_residual_energy",
            "valid",
        )
        missing = [key for key in required if key not in source]
        if missing:
            raise KeyError(f"{residual_path} is missing arrays: {missing}")
        residual_arrays = {key: source[key].copy() for key in required}
    with open(summary_path, "r") as f:
        summary = json.load(f)

    if not np.array_equal(arrays["times"], residual_arrays["times"]):
        raise ValueError("Anisotropy and residual time grids do not match")
    if not np.array_equal(arrays["labels"], residual_arrays["labels"]):
        raise ValueError("Anisotropy and residual labels do not match")
    if not np.array_equal(
        arrays["sample_indices"], residual_arrays["sample_indices"]
    ):
        raise ValueError("Anisotropy and residual sample indices do not match")

    times = arrays["times"]
    z0, z1 = arrays["z0"], arrays["z1"]
    directions = residual_arrays["residual_directions"]
    expected_direction_shape = (len(times), len(z0), *z0.shape[1:])
    if directions.shape != expected_direction_shape:
        raise ValueError(
            f"residual_directions shape={directions.shape}; "
            f"expected={expected_direction_shape}"
        )
    if z0.shape != z1.shape:
        raise ValueError(f"z0 shape={z0.shape} does not match z1 shape={z1.shape}")
    for key in (
        "residual_norm_squared",
        "alignment_ratio",
        "residual_aligned_gain",
        "propagated_residual_energy",
        "valid",
    ):
        expected = (len(times), len(z0))
        if residual_arrays[key].shape != expected:
            raise ValueError(
                f"{key} shape={residual_arrays[key].shape}; expected={expected}"
            )

    arrays.update(residual_arrays)
    arrays["summary"] = summary
    return arrays


@torch.inference_mode()
def evaluate_teacher_velocity_alignment(
    teacher: torch.nn.Module,
    arrays: Dict[str, np.ndarray],
    batch_size: int,
    compute_dtype,
    device: torch.device,
    min_velocity_norm: float,
) -> Dict[str, np.ndarray]:
    times = arrays["times"].astype(np.float64, copy=False)
    z0_cpu = torch.from_numpy(arrays["z0"]).float()
    z1_cpu = torch.from_numpy(arrays["z1"]).float()
    labels_cpu = torch.from_numpy(arrays["labels"]).long()
    residual_directions = torch.from_numpy(arrays["residual_directions"]).float()
    residual_norm_squared = torch.from_numpy(
        arrays["residual_norm_squared"]
    ).double()
    saved_valid = torch.from_numpy(arrays["valid"]).bool()

    num_times, num_states = len(times), len(z0_cpu)
    output_shape = (num_times, num_states)
    signed_cosine = torch.full(output_shape, torch.nan, dtype=torch.float64)
    cosine_squared = torch.full_like(signed_cosine, torch.nan)
    velocity_norm_squared = torch.full_like(signed_cosine, torch.nan)
    velocity_scale_coefficient = torch.full_like(signed_cosine, torch.nan)
    parallel_residual_norm_squared = torch.full_like(signed_cosine, torch.nan)
    perpendicular_residual_norm_squared = torch.full_like(signed_cosine, torch.nan)
    valid = torch.zeros(output_shape, dtype=torch.bool)
    velocity_directions = torch.zeros_like(residual_directions)

    for time_index, t_value in enumerate(times):
        for start in range(0, num_states, batch_size):
            end = min(start + batch_size, num_states)
            z0 = z0_cpu[start:end].to(device, non_blocking=True)
            z1 = z1_cpu[start:end].to(device, non_blocking=True)
            labels = labels_cpu[start:end].to(device, non_blocking=True)
            z_t = ((1.0 - float(t_value)) * z0 + float(t_value) * z1).float()
            time_batch = torch.full(
                (end - start,),
                float(t_value),
                device=device,
                dtype=torch.float32,
            )
            with _autocast_context(device, compute_dtype):
                velocity = teacher(z_t, time_batch, y=labels)
            velocity = velocity.float()
            velocity_norm = velocity.flatten(1).norm(dim=1)

            residual_direction = residual_directions[
                time_index, start:end
            ].to(device, non_blocking=True)
            residual_direction_norm = residual_direction.flatten(1).norm(dim=1)
            batch_saved_valid = saved_valid[time_index, start:end].to(device)
            batch_valid = (
                batch_saved_valid
                & torch.isfinite(velocity_norm)
                & torch.isfinite(residual_direction_norm)
                & (velocity_norm >= min_velocity_norm)
                & (residual_direction_norm > 0.0)
            )
            if not batch_valid.any():
                continue

            unit_velocity = torch.zeros_like(velocity)
            unit_residual = torch.zeros_like(residual_direction)
            velocity_view = velocity_norm[batch_valid].reshape(
                int(batch_valid.sum().item()), *([1] * (velocity.ndim - 1))
            )
            residual_view = residual_direction_norm[batch_valid].reshape(
                int(batch_valid.sum().item()),
                *([1] * (residual_direction.ndim - 1)),
            )
            unit_velocity[batch_valid] = velocity[batch_valid] / velocity_view
            unit_residual[batch_valid] = (
                residual_direction[batch_valid] / residual_view
            )
            cosine = (
                unit_residual[batch_valid] * unit_velocity[batch_valid]
            ).flatten(1).sum(dim=1).double().clamp(-1.0, 1.0)
            cosine2 = cosine.square()
            local_valid_indices = batch_valid.nonzero(as_tuple=False).flatten().cpu()
            global_valid_indices = local_valid_indices + start
            raw_residual_norm = residual_norm_squared[
                time_index, global_valid_indices
            ].sqrt().to(device)
            coefficient = (
                raw_residual_norm.double() * cosine / velocity_norm[batch_valid].double()
            )

            valid[time_index, global_valid_indices] = True
            signed_cosine[time_index, global_valid_indices] = cosine.cpu()
            cosine_squared[time_index, global_valid_indices] = cosine2.cpu()
            velocity_norm_squared[time_index, global_valid_indices] = (
                velocity_norm[batch_valid].double().square().cpu()
            )
            velocity_scale_coefficient[
                time_index, global_valid_indices
            ] = coefficient.cpu()
            batch_residual_norm_squared = residual_norm_squared[
                time_index, global_valid_indices
            ]
            parallel_residual_norm_squared[
                time_index, global_valid_indices
            ] = batch_residual_norm_squared * cosine2.cpu()
            perpendicular_residual_norm_squared[
                time_index, global_valid_indices
            ] = batch_residual_norm_squared * (1.0 - cosine2.cpu())
            velocity_directions[time_index, global_valid_indices] = (
                unit_velocity[batch_valid].cpu()
            )

    return {
        "valid": valid.numpy(),
        "signed_cosine": signed_cosine.numpy(),
        "cosine_squared": cosine_squared.numpy(),
        "teacher_velocity_norm_squared": velocity_norm_squared.numpy(),
        "velocity_scale_coefficient": velocity_scale_coefficient.numpy(),
        "parallel_residual_norm_squared": parallel_residual_norm_squared.numpy(),
        "perpendicular_residual_norm_squared": (
            perpendicular_residual_norm_squared.numpy()
        ),
        "teacher_velocity_directions": velocity_directions.numpy(),
    }


def summarize_timepoint(
    arrays: Dict[str, np.ndarray],
    results: Dict[str, np.ndarray],
    time_index: int,
) -> dict:
    valid = results["valid"][time_index]
    if int(valid.sum()) == 0:
        raise RuntimeError(f"No valid states at time index {time_index}")
    cosine = results["signed_cosine"][time_index, valid]
    cosine2 = results["cosine_squared"][time_index, valid]
    coefficient = results["velocity_scale_coefficient"][time_index, valid]
    velocity_norm2 = results["teacher_velocity_norm_squared"][time_index, valid]
    residual_norm2 = arrays["residual_norm_squared"][time_index, valid]
    parallel_norm2 = results["parallel_residual_norm_squared"][time_index, valid]
    perpendicular_norm2 = results[
        "perpendicular_residual_norm_squared"
    ][time_index, valid]
    gamma = arrays["alignment_ratio"][time_index, valid]
    residual_gain = arrays["residual_aligned_gain"][time_index, valid]
    propagated_energy = arrays["propagated_residual_energy"][time_index, valid]
    log_gamma = np.log(np.maximum(gamma, np.finfo(np.float64).tiny))
    log_gain = np.log(np.maximum(residual_gain, np.finfo(np.float64).tiny))
    log_energy = np.log(np.maximum(propagated_energy, np.finfo(np.float64).tiny))

    return {
        "num_states": int(valid.size),
        "num_valid_states": int(valid.sum()),
        "num_filtered_states": int(valid.size - valid.sum()),
        "signed_cosine_statistics": distribution_summary(cosine),
        "absolute_cosine_statistics": distribution_summary(np.abs(cosine)),
        "cosine_squared_statistics": distribution_summary(cosine2),
        "velocity_scale_coefficient_statistics": distribution_summary(coefficient),
        "teacher_velocity_norm_squared_statistics": distribution_summary(
            velocity_norm2
        ),
        "parallel_residual_norm_squared_statistics": distribution_summary(
            parallel_norm2
        ),
        "perpendicular_residual_norm_squared_statistics": distribution_summary(
            perpendicular_norm2
        ),
        "residual_norm_weighted_parallel_fraction": float(
            parallel_norm2.sum() / residual_norm2.sum()
        ),
        "fraction_cosine_squared_above_0_5": float((cosine2 > 0.5).mean()),
        "fraction_cosine_squared_above_0_9": float((cosine2 > 0.9).mean()),
        "fraction_cosine_squared_above_0_99": float((cosine2 > 0.99).mean()),
        "fraction_positive_signed_cosine": float((cosine > 0.0).mean()),
        "pearson_cosine_squared_vs_log_alignment_ratio": pearson_correlation(
            cosine2, log_gamma
        ),
        "spearman_cosine_squared_vs_alignment_ratio": spearman_correlation(
            cosine2, gamma
        ),
        "pearson_cosine_squared_vs_log_residual_gain": pearson_correlation(
            cosine2, log_gain
        ),
        "pearson_cosine_squared_vs_log_propagated_energy": pearson_correlation(
            cosine2, log_energy
        ),
    }


def save_outputs(
    output_dir: str,
    arrays: Dict[str, np.ndarray],
    results: Dict[str, np.ndarray],
    summaries: list,
    metadata: dict,
) -> None:
    times = arrays["times"].astype(np.float64, copy=False)
    np.savez_compressed(
        os.path.join(output_dir, "residual_teacher_velocity_alignment.npz"),
        times=times,
        sample_indices=arrays["sample_indices"],
        labels=arrays["labels"],
        residual_norm_squared=arrays["residual_norm_squared"],
        alignment_ratio=arrays["alignment_ratio"],
        residual_aligned_gain=arrays["residual_aligned_gain"],
        propagated_residual_energy=arrays["propagated_residual_energy"],
        **results,
    )
    with open(
        os.path.join(output_dir, "residual_teacher_velocity_alignment_summary.json"),
        "w",
    ) as f:
        json.dump({"metadata": metadata, "statistics": summaries}, f, indent=2)

    summary_rows = []
    for t_value, summary in zip(times, summaries):
        cosine = summary["signed_cosine_statistics"]
        cosine2 = summary["cosine_squared_statistics"]
        coefficient = summary["velocity_scale_coefficient_statistics"]
        summary_rows.append({
            "t": float(t_value),
            "num_valid_states": summary["num_valid_states"],
            "median_signed_cosine": cosine["median"],
            "q10_signed_cosine": cosine["q10"],
            "q90_signed_cosine": cosine["q90"],
            "median_cosine_squared": cosine2["median"],
            "q10_cosine_squared": cosine2["q10"],
            "q90_cosine_squared": cosine2["q90"],
            "fraction_cosine_squared_above_0_5": summary[
                "fraction_cosine_squared_above_0_5"
            ],
            "fraction_cosine_squared_above_0_9": summary[
                "fraction_cosine_squared_above_0_9"
            ],
            "fraction_cosine_squared_above_0_99": summary[
                "fraction_cosine_squared_above_0_99"
            ],
            "median_velocity_scale_coefficient": coefficient["median"],
            "q10_velocity_scale_coefficient": coefficient["q10"],
            "q90_velocity_scale_coefficient": coefficient["q90"],
            "residual_norm_weighted_parallel_fraction": summary[
                "residual_norm_weighted_parallel_fraction"
            ],
            "spearman_cosine_squared_vs_alignment_ratio": summary[
                "spearman_cosine_squared_vs_alignment_ratio"
            ],
        })
    with open(
        os.path.join(output_dir, "residual_teacher_velocity_alignment_summary.csv"),
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    state_rows = []
    for time_index, t_value in enumerate(times):
        for state_index in range(len(arrays["sample_indices"])):
            state_rows.append({
                "t": float(t_value),
                "state_index": state_index,
                "dataset_index": int(arrays["sample_indices"][state_index]),
                "label": int(arrays["labels"][state_index]),
                "valid": bool(results["valid"][time_index, state_index]),
                "signed_cosine": results["signed_cosine"][time_index, state_index],
                "cosine_squared": results["cosine_squared"][time_index, state_index],
                "teacher_velocity_norm_squared": results[
                    "teacher_velocity_norm_squared"
                ][time_index, state_index],
                "velocity_scale_coefficient": results[
                    "velocity_scale_coefficient"
                ][time_index, state_index],
                "residual_norm_squared": arrays["residual_norm_squared"][
                    time_index, state_index
                ],
                "parallel_residual_norm_squared": results[
                    "parallel_residual_norm_squared"
                ][time_index, state_index],
                "perpendicular_residual_norm_squared": results[
                    "perpendicular_residual_norm_squared"
                ][time_index, state_index],
                "alignment_ratio": arrays["alignment_ratio"][
                    time_index, state_index
                ],
                "residual_aligned_gain": arrays["residual_aligned_gain"][
                    time_index, state_index
                ],
                "propagated_residual_energy": arrays[
                    "propagated_residual_energy"
                ][time_index, state_index],
            })
    with open(
        os.path.join(output_dir, "residual_teacher_velocity_alignment_states.csv"),
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(state_rows[0].keys()))
        writer.writeheader()
        writer.writerows(state_rows)

    plot_diagnostics(output_dir, times, arrays, results, summaries)


def plot_diagnostics(
    output_dir: str,
    times: np.ndarray,
    arrays: Dict[str, np.ndarray],
    results: Dict[str, np.ndarray],
    summaries: list,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cosine2_median = [row["cosine_squared_statistics"]["median"] for row in summaries]
    cosine2_q10 = [row["cosine_squared_statistics"]["q10"] for row in summaries]
    cosine2_q90 = [row["cosine_squared_statistics"]["q90"] for row in summaries]
    signed_median = [row["signed_cosine_statistics"]["median"] for row in summaries]
    signed_q10 = [row["signed_cosine_statistics"]["q10"] for row in summaries]
    signed_q90 = [row["signed_cosine_statistics"]["q90"] for row in summaries]
    coefficient_median = [
        row["velocity_scale_coefficient_statistics"]["median"] for row in summaries
    ]
    coefficient_q10 = [
        row["velocity_scale_coefficient_statistics"]["q10"] for row in summaries
    ]
    coefficient_q90 = [
        row["velocity_scale_coefficient_statistics"]["q90"] for row in summaries
    ]

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(times, cosine2_median, marker="o", label=r"median $\cos^2$")
    axes[0, 0].fill_between(times, cosine2_q10, cosine2_q90, alpha=0.25, label="q10–q90")
    axes[0, 0].set_ylim(-0.03, 1.03)
    axes[0, 0].set_xlabel("t")
    axes[0, 0].set_ylabel(r"$\cos^2(\rho,v_{teacher})$")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(times, signed_median, marker="o", label="median signed cosine")
    axes[0, 1].fill_between(times, signed_q10, signed_q90, alpha=0.25, label="q10–q90")
    axes[0, 1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[0, 1].set_ylim(-1.03, 1.03)
    axes[0, 1].set_xlabel("t")
    axes[0, 1].set_ylabel("signed cosine")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(times, coefficient_median, marker="o", label="median c")
    axes[1, 0].fill_between(
        times, coefficient_q10, coefficient_q90, alpha=0.25, label="q10–q90"
    )
    axes[1, 0].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1, 0].set_xlabel("t")
    axes[1, 0].set_ylabel(r"$c=\langle\rho,v\rangle/\|v\|^2$")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    for time_index, t_value in enumerate(times):
        valid = results["valid"][time_index]
        axes[1, 1].scatter(
            results["cosine_squared"][time_index, valid],
            arrays["alignment_ratio"][time_index, valid],
            alpha=0.7,
            label=f"t={t_value:.5g}",
        )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel(r"$\cos^2(\rho,v_{teacher})$")
    axes[1, 1].set_ylabel(r"residual alignment ratio $\gamma$")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    figure.suptitle("Residual alignment with teacher velocity")
    figure.tight_layout()
    figure.savefig(
        os.path.join(output_dir, "residual_teacher_velocity_alignment.png"),
        dpi=180,
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare saved residual directions with teacher velocity"
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional teacher checkpoint override when metadata path moved",
    )
    parser.add_argument("--weight-key", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--compute-dtype", default=None)
    parser.add_argument("--min-velocity-norm", type=float, default=1e-8)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.min_velocity_norm <= 0.0:
        raise ValueError("--min-velocity-norm must be positive")

    started_at = datetime.now(timezone.utc)
    start = perf_counter()
    results_dir = os.path.abspath(args.results_dir)
    arrays = load_existing_results(results_dir)
    source_metadata = arrays["summary"]["metadata"]
    checkpoint_value = args.checkpoint or source_metadata["teacher_checkpoint"]
    checkpoint_path = resolve_checkpoint_path({"ckpt_path": checkpoint_value})
    weight_key = args.weight_key or source_metadata.get("teacher_weights", "ema")
    compute_dtype_name = args.compute_dtype or source_metadata.get(
        "compute_dtype", "float32"
    )
    compute_dtype = parse_compute_dtype(compute_dtype_name)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    print(f"Loading frozen teacher: {checkpoint_path} [{weight_key}]")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if weight_key not in checkpoint:
        raise KeyError(
            f"Checkpoint has no '{weight_key}' weights; "
            f"available keys={list(checkpoint.keys())}"
        )
    model_args = resolve_model_args(
        checkpoint.get("args"), source_metadata.get("teacher_model", {})
    )
    teacher_weights = checkpoint[weight_key]
    del checkpoint
    teacher = SiT_models[model_args.model](
        input_size=model_args.image_size // 8,
        num_classes=model_args.num_classes,
    ).to(device)
    teacher.load_state_dict(teacher_weights)
    del teacher_weights
    teacher.eval()
    teacher.requires_grad_(False)
    synchronize_device(device)
    initialization_seconds = perf_counter() - start

    evaluation_start = perf_counter()
    results = evaluate_teacher_velocity_alignment(
        teacher=teacher,
        arrays=arrays,
        batch_size=args.batch_size,
        compute_dtype=compute_dtype,
        device=device,
        min_velocity_norm=args.min_velocity_norm,
    )
    synchronize_device(device)
    evaluation_seconds = perf_counter() - evaluation_start
    summaries = []
    for time_index, t_value in enumerate(arrays["times"]):
        summary = summarize_timepoint(arrays, results, time_index)
        summary["t"] = float(t_value)
        summaries.append(summary)
        print(
            f"t={t_value:.5f} valid={summary['num_valid_states']} "
            f"median_cos2={summary['cosine_squared_statistics']['median']:.6g} "
            f"cos2>0.9={summary['fraction_cosine_squared_above_0_9']:.1%} "
            f"median_c={summary['velocity_scale_coefficient_statistics']['median']:.6g}"
        )

    metadata = {
        "source_results_dir": results_dir,
        "teacher_checkpoint": os.path.abspath(checkpoint_path),
        "teacher_weights": weight_key,
        "teacher_model": vars(model_args),
        "compute_dtype": compute_dtype_name,
        "batch_size": args.batch_size,
        "min_velocity_norm": args.min_velocity_norm,
        "residual_definition": source_metadata.get("residual_definition"),
        "cosine_definition": (
            "signed_cosine=<unit_residual,unit_teacher_velocity>; "
            "cosine_squared=signed_cosine^2"
        ),
        "velocity_scale_coefficient_definition": (
            "c=<raw_residual,teacher_velocity>/||teacher_velocity||^2"
        ),
        "requires_ode_rollout": False,
        "requires_student_model": False,
    }
    save_start = perf_counter()
    save_outputs(results_dir, arrays, results, summaries, metadata)
    save_seconds = perf_counter() - save_start
    finished_at = datetime.now(timezone.utc)
    total_seconds = perf_counter() - start
    timing = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "initialization_seconds": initialization_seconds,
        "teacher_forward_seconds": evaluation_seconds,
        "save_seconds": save_seconds,
        "total_wall_seconds": total_seconds,
        "single_gpu_wall_hours": total_seconds / 3600.0,
    }
    with open(
        os.path.join(results_dir, "residual_teacher_velocity_alignment_timing.json"),
        "w",
    ) as f:
        json.dump(timing, f, indent=2)
    print(f"Saved teacher-velocity alignment results to {results_dir}")
    print(f"Total elapsed: {format_duration(total_seconds)}")


if __name__ == "__main__":
    main()
