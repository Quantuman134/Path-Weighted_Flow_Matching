"""
exp_w_avg_finite_difference.py -- Randomized finite-difference estimation of
                                  w_avg(t) for a pretrained SiT-XL/2 flow.

Given the pretrained deterministic flow map  F_{t->1}(z)  produced by
SiT-XL/2 + fixed-step Euler ODE, we estimate the average endpoint-error
amplification

    w_avg(t) = E_{z_t}[ (1/d) tr( Phi(1,t;z_t)^T Phi(1,t;z_t) ) ]

using randomized Rademacher probes and paired central-difference JVPs:

    Phi(1,t;z_t) q  ~=  ( F_{t->1}(z_t + delta q) - F_{t->1}(z_t - delta q) ) / (2 delta)

Two modes are supported (selected by cfg["experiment"]["mode"]):

    stability : Experiment A -- sweep eta in {1e-5,1e-4,1e-3,1e-2} at
                t in {0, 0.5, 0.9} to pick a stable perturbation scale.

    w_avg     : Experiment B -- estimate w_avg(t) on a 16-timestep grid at
                the chosen eta and compare with A(t)^2.

Both modes share the same underlying primitives: cache 50-step trajectories,
then re-run partial (50 - s_j)-step Euler solves from a perturbed state.

Usage (single-GPU):
    python exp_w_avg_finite_difference.py \
        --config configs/exp_w_avg_finite_difference_config.yaml --device cuda:0

Usage (multi-GPU, base trajectories split evenly across ranks):
    torchrun --nproc_per_node=8 exp_w_avg_finite_difference.py \
        --config configs/exp_w_avg_finite_difference_config.yaml
"""

import argparse
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import yaml

_SIT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SIT_DIR not in sys.path:
    sys.path.insert(0, _SIT_DIR)

from models import SiT_models
from download import find_model


# ------------------------------------------------------------------ constants

LATENT_C = 4
LATENT_H = 32
LATENT_W = 32
LATENT_D = LATENT_C * LATENT_H * LATENT_W          # 4096


# ------------------------------------------------------------------ distrib.

def setup_dist(default_device: str):
    """Initialize NCCL when torchrun sets RANK/WORLD_SIZE, else fall back."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = default_device
    return rank, world_size, device


def barrier(world_size: int):
    if world_size > 1:
        dist.barrier()


def all_reduce_sum(t: torch.Tensor, world_size: int) -> torch.Tensor:
    """In-place all-reduce (sum) when distributed, identity otherwise."""
    if world_size > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


# ------------------------------------------------------------------ config

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ model

def build_pretrained_sit_xl_2(device: str, num_classes: int = 1000,
                              image_size: int = 256):
    """Instantiate SiT-XL/2 (learn_sigma=True) and load the pretrained weights.

    Returns model in eval mode on device, in fp32.
    """
    latent_size = image_size // 8            # 32
    model = SiT_models["SiT-XL/2"](
        input_size=latent_size,
        num_classes=num_classes,
        learn_sigma=True,                     # matches the 256x256 pretrained ckpt
    ).to(device).float()

    ckpt_path = f"SiT-XL-2-{image_size}x{image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ------------------------------------------------------------------ Euler ODE

@torch.no_grad()
def euler_step(model, z: torch.Tensor, t_val: float, dt: float,
               y: torch.Tensor) -> torch.Tensor:
    """One Euler step of the velocity ODE: z_{t+dt} = z_t + dt * v_theta(z_t, t).

    Uses forward() (returns only the velocity slice; learn_sigma channels
    are discarded internally). t is broadcast to (B,) as required.
    """
    t = torch.full((z.shape[0],), float(t_val), device=z.device, dtype=z.dtype)
    v = model(z, t, y)
    return z + dt * v


@torch.no_grad()
def integrate_partial(model, z_start: torch.Tensor, y: torch.Tensor,
                      s_start: int, num_steps: int) -> torch.Tensor:
    """Continue Euler integration from solver-index s_start to num_steps.

    That is, we start at t = s_start / num_steps and end at t = 1, taking
    (num_steps - s_start) Euler steps of size 1/num_steps. When
    s_start == num_steps, this is the identity map (endpoint is s_start).
    """
    if s_start >= num_steps:
        return z_start
    dt = 1.0 / num_steps
    z = z_start
    for k in range(s_start, num_steps):
        t_val = k * dt
        z = euler_step(model, z, t_val, dt, y)
    return z


# ------------------------------------------------------------------ probes

def rademacher_directions(n: int, k: int, shape, device, generator):
    """Return (n, k, *shape) tensor of unit-norm Rademacher probes.

    Each probe is  q = r / sqrt(d),  r_j in {-1,+1} uniformly.  Then
    ||q||_2 = 1 and E[q q^T] = I / d, the trace-per-dim probe.
    """
    d = int(np.prod(shape))
    r = torch.empty((n, k, *shape), device=device, dtype=torch.float32)
    # bernoulli(0.5) -> {0,1}; map to {-1,+1}.
    r.bernoulli_(0.5, generator=generator).mul_(2.0).sub_(1.0)
    r.div_(math.sqrt(d))
    return r


# ------------------------------------------------------------------ scale c(t)

def per_coord_rms(states: torch.Tensor) -> float:
    """Centered per-coordinate RMS of a stack of (N, *shape) latents.

        c = sqrt( (1/(N*d)) * sum_i ||z_i - mean||^2 ).

    Computed in float64 for stability on very large N.
    """
    x = states.reshape(states.shape[0], -1).to(torch.float64)
    mu = x.mean(dim=0, keepdim=True)
    diff = x - mu
    return float(torch.sqrt((diff * diff).mean()).item())


# ------------------------------------------------------------------ A(t)^2

def analytic_A_squared(t: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Vanilla parameterized weighting A(t)^2 = lam^2 / ((1-t)^2 + lam^2 t^2).

    For lam = 1 this reduces to 1 / ((1-t)^2 + t^2). This function corresponds
    to the LossSpace.VANILLA_WEIGHTING_V weight in transport/transport.py.
    """
    return (lam ** 2) / ((1.0 - t) ** 2 + (lam ** 2) * (t ** 2) + 1e-12)


# ------------------------------------------------------------------ base traj

@torch.no_grad()
def sample_base_trajectories(model, n_local: int, num_classes: int,
                             num_solver_steps: int, device: str, seed: int,
                             rank: int, world_size: int, batch_size: int):
    """Generate the local rank's N_local base trajectories.

    Returns:
        cache: list of length (num_solver_steps + 1) of CPU float32
               tensors (n_local, C, H, W). Each is z_{k/num_solver_steps}.
        labels: (n_local,) int64 tensor of class labels used for the runs.

    Per-rank noise seed guarantees distinct z0 across ranks while remaining
    deterministic. Class labels are drawn uniformly at random.
    """
    torch.manual_seed(seed * (world_size + 1) + rank)
    rng = np.random.default_rng(seed * (world_size + 1) + rank + 13)

    labels_np = rng.integers(0, num_classes, size=(n_local,), dtype=np.int64)
    labels = torch.from_numpy(labels_np).to(device)

    # Allocate the caches (num_steps + 1) tensors on CPU up front.
    cache = [torch.empty((n_local, LATENT_C, LATENT_H, LATENT_W),
                         dtype=torch.float32)
             for _ in range(num_solver_steps + 1)]

    for start in range(0, n_local, batch_size):
        end = min(start + batch_size, n_local)
        z = torch.randn((end - start, LATENT_C, LATENT_H, LATENT_W),
                        device=device, dtype=torch.float32)
        y_b = labels[start:end]
        cache[0][start:end] = z.detach().to("cpu")
        dt = 1.0 / num_solver_steps
        for k in range(num_solver_steps):
            t_val = k * dt
            z = euler_step(model, z, t_val, dt, y_b)
            cache[k + 1][start:end] = z.detach().to("cpu")
        if rank == 0:
            print(f"\r  Base trajectories: {end}/{n_local} (rank 0)",
                  end="", flush=True)
    if rank == 0:
        print()

    return cache, labels


# ------------------------------------------------------------------ probe run

@torch.no_grad()
def measure_amplification(model, z_state: torch.Tensor, labels: torch.Tensor,
                          s_start: int, num_solver_steps: int,
                          delta_scalar: float, K: int, device: str,
                          probe_batch: int, generator) -> torch.Tensor:
    """Compute per-probe amplification g_{i,k} = ||( F(z+dq) - F(z-dq) )/(2d)||^2
    for one cached state z_state of shape (n_local, C, H, W), with K
    Rademacher probes per state and paired central differences.

    Returns:
        g: (n_local, K) float64 CPU tensor of ||Phi(1,t) q||^2 estimates.
    """
    n_local = z_state.shape[0]
    g = torch.empty((n_local, K), dtype=torch.float64)
    if n_local == 0:
        return g

    d = LATENT_D
    inv_2d = 1.0 / (2.0 * delta_scalar)

    for k_idx in range(K):
        # Draw one probe direction per base state.
        q = rademacher_directions(n_local, 1, (LATENT_C, LATENT_H, LATENT_W),
                                  device, generator).squeeze(1)  # (n_local, C, H, W)
        # Paired trajectories are batched together: (2*n_local, C, H, W).
        for start in range(0, n_local, probe_batch):
            end = min(start + probe_batch, n_local)
            z_plus  = (z_state[start:end].to(device) + delta_scalar * q[start:end])
            z_minus = (z_state[start:end].to(device) - delta_scalar * q[start:end])
            z_pair = torch.cat([z_plus, z_minus], dim=0)
            y_pair = torch.cat([labels[start:end], labels[start:end]], dim=0)

            z_out = integrate_partial(model, z_pair, y_pair,
                                      s_start, num_solver_steps)
            f_plus, f_minus = z_out.chunk(2, dim=0)
            diff = (f_plus - f_minus) * inv_2d   # (b, C, H, W)
            # ||.||^2 across all latent dims -> (b,).
            sq = diff.reshape(diff.shape[0], -1).to(torch.float64)
            g[start:end, k_idx] = (sq * sq).sum(dim=1).cpu()
    return g


# ------------------------------------------------------------------ stats

def bootstrap_ci(values: np.ndarray, n_boot: int = 2000,
                 seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI over the FIRST axis of a (N, ...) array.

    Returns (mean, ci_low, ci_high), each of shape (values.shape[1:],).
    """
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    mean_val = values.mean(axis=0)
    # Draw bootstrap resamples; store their means only.
    means = np.empty((n_boot,) + values.shape[1:], dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = values[idx].mean(axis=0)
    lo = np.quantile(means, alpha / 2, axis=0)
    hi = np.quantile(means, 1 - alpha / 2, axis=0)
    return mean_val, lo, hi


# ------------------------------------------------------------------ plotting

def _lazy_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_stability(mu, disagreement, cv, etas, times, out_dir: str):
    """Plot A (mu vs eta), Plot B (paired disagreement), Plot C (CV)."""
    plt = _lazy_plt()

    # Plot A: mean estimate vs eta, one curve per timestep.
    plt.figure(figsize=(8, 5))
    for j, t in enumerate(times):
        plt.plot(etas, mu[j], marker="o", label=f"t={t:g}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("perturbation ratio eta")
    plt.ylabel("mean amplification ||Phi q||^2")
    plt.title("Stability check: mean amplification vs eta")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stability_plotA_mean_vs_eta.png"), dpi=140)
    plt.close()

    # Plot B: paired adjacent-scale median relative disagreement.
    plt.figure(figsize=(8, 5))
    for j, t in enumerate(times):
        # disagreement[j] has shape (len(etas)-1,) between adjacent etas.
        plt.plot(etas[:-1], disagreement[j], marker="s", label=f"t={t:g}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("perturbation ratio eta (adjacent pair)")
    plt.ylabel("median relative disagreement")
    plt.title("Stability check: adjacent-eta paired disagreement")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stability_plotB_disagreement.png"), dpi=140)
    plt.close()

    # Plot C: coefficient of variation across probes.
    plt.figure(figsize=(8, 5))
    for j, t in enumerate(times):
        plt.plot(etas, cv[j], marker="^", label=f"t={t:g}")
    plt.xscale("log")
    plt.xlabel("perturbation ratio eta")
    plt.ylabel("CV of ||Phi q||^2 across probes")
    plt.title("Stability check: probe-level coefficient of variation")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stability_plotC_cv.png"), dpi=140)
    plt.close()


def plot_wavg(t_grid, mean_w, ci_lo, ci_hi, A_sq, out_dir: str,
              lam: float = 1.0):
    """Plot E (raw w_avg with CI), Plot F (raw vs A(t)^2, both normalized)."""
    plt = _lazy_plt()

    # Plot E: raw estimate with 95% bootstrap CI.
    plt.figure(figsize=(9, 5))
    plt.plot(t_grid, mean_w, marker="o", linewidth=2, label="w_avg(t) est.")
    plt.fill_between(t_grid, ci_lo, ci_hi, alpha=0.25,
                     label="95% bootstrap CI")
    plt.xlabel("t")
    plt.ylabel("w_avg(t)")
    plt.title("Empirical w_avg(t) from Rademacher central-difference probes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "wavg_plotE_raw.png"), dpi=140)
    plt.close()

    # Plot F: raw vs A(t)^2 (unnormalized) + mean-1 normalization overlay.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(t_grid, mean_w, marker="o", label="w_avg(t) est.")
    axes[0].plot(t_grid, A_sq,   marker="s", label=f"A(t)^2 (lam={lam:g})")
    axes[0].fill_between(t_grid, ci_lo, ci_hi, alpha=0.25)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("value")
    axes[0].set_yscale("log")
    axes[0].set_title("w_avg(t) vs A(t)^2 (log-y)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    # Mean-1 normalised comparison (both curves rescaled to mean = 1 over grid).
    def _norm(x):
        m = float(np.mean(x))
        return np.asarray(x) / (m + 1e-12)

    axes[1].plot(t_grid, _norm(mean_w), marker="o", label="norm w_avg(t)")
    axes[1].plot(t_grid, _norm(A_sq),   marker="s", label=f"norm A(t)^2 (lam={lam:g})")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("normalized value")
    axes[1].set_title("Mean-1 normalized comparison")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "wavg_plotF_shape.png"), dpi=140)
    plt.close()


# ------------------------------------------------------------------ mode: A

def run_stability(cfg, model, cache_local, labels_local, device, rank,
                  world_size, out_dir, seed, num_solver_steps,
                  probe_batch, K):
    """Experiment A: sweep eta at a few timesteps."""
    stab_cfg = cfg["stability"]
    eta_list = list(stab_cfg["etas"])
    stab_times = list(stab_cfg["times"])
    # Solver-index of each requested time. Only exact multiples of 1/T supported.
    s_list = [int(round(t * num_solver_steps)) for t in stab_times]

    # Compute c(t) globally by all-reducing (sum, sq-sum, count) per state.
    c_of_t = {}
    for s in s_list:
        z_local = cache_local[s].to(device, dtype=torch.float64)
        n_local = z_local.shape[0]
        # Compute local sum, sum_sq per coordinate (flattened) then all-reduce.
        z_flat = z_local.reshape(n_local, -1)
        loc_sum   = z_flat.sum(dim=0)
        loc_count = torch.tensor([n_local], device=device, dtype=torch.float64)
        loc_sq    = (z_flat * z_flat).sum(dim=0)
        all_reduce_sum(loc_sum,   world_size)
        all_reduce_sum(loc_count, world_size)
        all_reduce_sum(loc_sq,    world_size)
        n_total = float(loc_count.item())
        mu = loc_sum / n_total
        # (1/(N*d)) * sum ||z - mu||^2 = mean(sq_sum/N - mu^2)
        var_per_coord = (loc_sq / n_total) - mu * mu
        c_val = float(torch.sqrt(var_per_coord.clamp_min(0.0).mean()).item())
        c_of_t[s] = c_val
        if rank == 0:
            print(f"  c(t={s / num_solver_steps:.3f}) = {c_val:.6f}")

    # For each (t, eta): collect per-probe amplification g_{i,k,eta}(t).
    #
    # Storage: dict[(s, eta_idx)] -> (n_local, K) float64 CPU tensor.
    per_probe = {}
    generator = torch.Generator(device=device).manual_seed(
        seed * 1000003 + rank * 7 + 1
    )

    total_pairs = len(s_list) * len(eta_list)
    done = 0
    t_start = time.time()

    for j, s in enumerate(s_list):
        for e_idx, eta in enumerate(eta_list):
            delta = eta * c_of_t[s] * math.sqrt(LATENT_D)
            g_ik = measure_amplification(
                model=model,
                z_state=cache_local[s],
                labels=labels_local,
                s_start=s,
                num_solver_steps=num_solver_steps,
                delta_scalar=delta,
                K=K,
                device=device,
                probe_batch=probe_batch,
                generator=generator,
            )
            per_probe[(s, e_idx)] = g_ik
            done += 1
            if rank == 0:
                elapsed = time.time() - t_start
                print(f"  [stability] t={s/num_solver_steps:.3f}"
                      f"  eta={eta:.0e}  delta={delta:.3e}"
                      f"  mean_local={float(g_ik.mean()):.4e}"
                      f"  ({done}/{total_pairs} pairs, elapsed={elapsed:.0f}s)")

    # Aggregate across ranks: for each (s, eta) compute global mean and CV.
    mu_arr    = np.zeros((len(s_list), len(eta_list)), dtype=np.float64)
    cv_arr    = np.zeros_like(mu_arr)
    disagree  = np.zeros((len(s_list), len(eta_list) - 1), dtype=np.float64)

    for j, s in enumerate(s_list):
        # First pass: means/variances.
        for e_idx in range(len(eta_list)):
            g_local = per_probe[(s, e_idx)].to(device, dtype=torch.float64)
            loc_sum   = g_local.sum().reshape(1)
            loc_sq    = (g_local * g_local).sum().reshape(1)
            loc_count = torch.tensor([g_local.numel()], device=device,
                                     dtype=torch.float64)
            all_reduce_sum(loc_sum,   world_size)
            all_reduce_sum(loc_sq,    world_size)
            all_reduce_sum(loc_count, world_size)
            n_total = float(loc_count.item())
            mean_v = float(loc_sum.item()) / n_total
            var_v  = float(loc_sq.item()) / n_total - mean_v * mean_v
            mu_arr[j, e_idx] = mean_v
            cv_arr[j, e_idx] = math.sqrt(max(var_v, 0.0)) / (mean_v + 1e-30)

        # Second pass: paired adjacent-eta disagreement (median relative diff).
        # For median we need the full sample; use gather_object.
        for e_idx in range(len(eta_list) - 1):
            a = per_probe[(s, e_idx)].reshape(-1).numpy()
            b = per_probe[(s, e_idx + 1)].reshape(-1).numpy()
            # relative diff |a - b| / mean(a, b)
            avg = 0.5 * (a + b)
            rel = np.abs(a - b) / (avg + 1e-30)

            if world_size > 1:
                gathered = [None] * world_size if rank == 0 else None
                dist.gather_object(rel, gathered, dst=0)
                if rank == 0:
                    all_rel = np.concatenate(gathered, axis=0)
                    disagree[j, e_idx] = float(np.median(all_rel))
            else:
                disagree[j, e_idx] = float(np.median(rel))

    if rank == 0:
        # Broadcast disagree from rank 0 to all ranks not needed; we only save it.
        result = {
            "mode": "stability",
            "num_solver_steps": num_solver_steps,
            "times": stab_times,
            "solver_indices": s_list,
            "etas": eta_list,
            "c_of_t": {str(s / num_solver_steps): c_of_t[s] for s in s_list},
            "mean_amplification": mu_arr.tolist(),   # (T, E)
            "cv_across_probes":   cv_arr.tolist(),
            "median_rel_disagreement_adjacent_eta": disagree.tolist(),  # (T, E-1)
        }
        with open(os.path.join(out_dir, "stability_results.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved -> {os.path.join(out_dir, 'stability_results.json')}")

        plot_stability(
            mu=mu_arr,
            disagreement=disagree,
            cv=cv_arr,
            etas=eta_list,
            times=stab_times,
            out_dir=out_dir,
        )

        # Suggest an eta by picking the largest one with adjacent disagreement
        # below the plateau threshold at every tested time.
        threshold = float(cfg["stability"].get("plateau_threshold", 0.05))
        candidate = None
        for e_idx in range(len(eta_list) - 1, 0, -1):
            ok = all(disagree[j, e_idx - 1] < threshold for j in range(len(s_list)))
            if ok:
                candidate = eta_list[e_idx]
                break
        print(f"[stability] suggested eta_star = {candidate}"
              f"  (threshold {threshold:g})")

    barrier(world_size)


# ------------------------------------------------------------------ mode: B

def run_wavg(cfg, model, cache_local, labels_local, device, rank,
             world_size, out_dir, seed, num_solver_steps, probe_batch, K):
    """Experiment B: 16-timestep w_avg(t) estimate at fixed eta_star."""
    wcfg = cfg["wavg"]
    eta_star = float(wcfg["eta_star"])
    s_grid = list(wcfg["solver_indices"])   # 16 checkpoints in [0, num_solver_steps]
    assert num_solver_steps in s_grid, \
        f"solver_indices must include {num_solver_steps} (t=1)."

    t_grid = np.array([s / num_solver_steps for s in s_grid], dtype=np.float64)

    # c(t) at every measured timestep (same all-reduce pattern as Experiment A).
    c_of_t = {}
    for s in s_grid:
        z_local = cache_local[s].to(device, dtype=torch.float64)
        n_local = z_local.shape[0]
        z_flat = z_local.reshape(n_local, -1)
        loc_sum   = z_flat.sum(dim=0)
        loc_count = torch.tensor([n_local], device=device, dtype=torch.float64)
        loc_sq    = (z_flat * z_flat).sum(dim=0)
        all_reduce_sum(loc_sum,   world_size)
        all_reduce_sum(loc_count, world_size)
        all_reduce_sum(loc_sq,    world_size)
        n_total = float(loc_count.item())
        mu = loc_sum / n_total
        var_per_coord = (loc_sq / n_total) - mu * mu
        c_of_t[s] = float(torch.sqrt(var_per_coord.clamp_min(0.0).mean()).item())

    if rank == 0:
        print("  c(t) values along the grid:")
        for s in s_grid:
            print(f"    t={s / num_solver_steps:.3f}   c={c_of_t[s]:.6f}")

    # Per-timestep amplification per base trajectory (averaged over K probes).
    #
    # Collect per-trajectory means g_i(t) = (1/K) sum_k g_{i,k,eta_star}(t),
    # which are the natural bootstrap unit. Layout: (n_local, T) float64.
    per_traj_mean_local = torch.zeros(
        (labels_local.shape[0], len(s_grid)), dtype=torch.float64
    )

    generator = torch.Generator(device=device).manual_seed(
        seed * 1000033 + rank * 11 + 7
    )
    total = len(s_grid)
    t_start = time.time()

    for j, s in enumerate(s_grid):
        if s == num_solver_steps:
            # Identity map: g_{i,k} = ||q||^2 = 1 exactly. Skip the solve.
            per_traj_mean_local[:, j] = 1.0
            if rank == 0:
                print(f"  [wavg] t=1.000  identity-map sanity check "
                      f"(w_avg(1)=1 exact)")
            continue

        delta = eta_star * c_of_t[s] * math.sqrt(LATENT_D)
        g_ik = measure_amplification(
            model=model,
            z_state=cache_local[s],
            labels=labels_local,
            s_start=s,
            num_solver_steps=num_solver_steps,
            delta_scalar=delta,
            K=K,
            device=device,
            probe_batch=probe_batch,
            generator=generator,
        )
        per_traj_mean_local[:, j] = g_ik.mean(dim=1)
        if rank == 0:
            elapsed = time.time() - t_start
            print(f"  [wavg] t={s / num_solver_steps:.3f}"
                  f"  eta={eta_star:.0e}  delta={delta:.3e}"
                  f"  mean_local={float(g_ik.mean()):.4e}"
                  f"  ({j + 1}/{total}, elapsed={elapsed:.0f}s)")

    # Gather all per-trajectory means to rank 0 for bootstrap.
    if world_size > 1:
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(per_traj_mean_local.numpy(), gathered, dst=0)
        if rank == 0:
            per_traj_mean_all = np.concatenate(gathered, axis=0)
    else:
        per_traj_mean_all = per_traj_mean_local.numpy()

    if rank == 0:
        # Bootstrap CI over base trajectories.
        n_boot = int(cfg["wavg"].get("bootstrap_iters", 2000))
        mean_w, ci_lo, ci_hi = bootstrap_ci(
            per_traj_mean_all, n_boot=n_boot, seed=seed
        )
        median_w = np.median(per_traj_mean_all, axis=0)
        q25 = np.quantile(per_traj_mean_all, 0.25, axis=0)
        q75 = np.quantile(per_traj_mean_all, 0.75, axis=0)
        cv_w = per_traj_mean_all.std(axis=0) / (mean_w + 1e-30)

        # Compare with A(t)^2.
        lam = float(cfg["wavg"].get("lam", 1.0))
        A_sq = analytic_A_squared(t_grid, lam=lam)

        # Correlation and mean-1 shape ratio R(t) = norm w / norm A^2.
        norm_w = mean_w / (mean_w.mean() + 1e-30)
        norm_A = A_sq / (A_sq.mean() + 1e-30)
        R_t = norm_w / (norm_A + 1e-30)
        pearson = float(np.corrcoef(mean_w, A_sq)[0, 1])

        result = {
            "mode": "wavg",
            "num_solver_steps": num_solver_steps,
            "eta_star": eta_star,
            "solver_indices": s_grid,
            "t_grid": t_grid.tolist(),
            "c_of_t": {str(s / num_solver_steps): c_of_t[s] for s in s_grid},
            "w_avg_mean": mean_w.tolist(),
            "w_avg_ci95_low":  ci_lo.tolist(),
            "w_avg_ci95_high": ci_hi.tolist(),
            "w_avg_median":  median_w.tolist(),
            "w_avg_q25":     q25.tolist(),
            "w_avg_q75":     q75.tolist(),
            "w_avg_cv_trajectories": cv_w.tolist(),
            "A_squared":     A_sq.tolist(),
            "A_squared_lam": lam,
            "normalized_ratio_R_t": R_t.tolist(),
            "pearson_corr_with_A_sq": pearson,
            "num_base_trajectories_total": int(per_traj_mean_all.shape[0]),
            "K_probes_per_state": K,
        }

        with open(os.path.join(out_dir, "wavg_results.json"), "w") as f:
            json.dump(result, f, indent=2)
        np.savez(
            os.path.join(out_dir, "wavg_per_trajectory_means.npz"),
            t_grid=t_grid,
            per_trajectory_mean=per_traj_mean_all,
        )
        print(f"Saved -> {os.path.join(out_dir, 'wavg_results.json')}")
        print(f"Pearson corr( w_avg(t) , A(t)^2 ) = {pearson:.4f}")

        plot_wavg(
            t_grid=t_grid,
            mean_w=mean_w,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            A_sq=A_sq,
            out_dir=out_dir,
            lam=lam,
        )

    barrier(world_size)


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(
        description="Finite-difference estimation of w_avg(t) on pretrained SiT-XL/2"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None,
                        help="Single-GPU device (overrides config, e.g. cuda:0)")
    cli = parser.parse_args()

    cfg = load_config(cli.config)

    default_device = cli.device or cfg.get("device", "cuda:0")
    rank, world_size, device = setup_dist(default_device)

    # Enforce FP32 (config option; kept for clarity).
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # ── experiment dir on rank 0 ─────────────────────────────────────────────
    exp_dir = os.path.join(
        cfg["experiment"]["experiments_base_dir"],
        cfg["experiment"]["name"],
    )
    if rank == 0:
        os.makedirs(exp_dir, exist_ok=True)
        shutil.copy(cli.config, os.path.join(exp_dir, "config.yaml"))
        print(f"Experiment dir : {exp_dir}")
        print(f"Rank {rank}/{world_size} on {device}")
    barrier(world_size)

    # ── model ────────────────────────────────────────────────────────────────
    mode = cfg["experiment"]["mode"]
    assert mode in ("stability", "wavg"), \
        f"experiment.mode must be 'stability' or 'wavg', got {mode!r}"

    if rank == 0:
        print(f"\nBuilding pretrained SiT-XL/2 (FP32) ...")
    model = build_pretrained_sit_xl_2(
        device=device,
        num_classes=int(cfg["model"].get("num_classes", 1000)),
        image_size=int(cfg["model"].get("image_size", 256)),
    )

    # ── budget ───────────────────────────────────────────────────────────────
    num_solver_steps = int(cfg["solver"]["num_steps"])
    K = int(cfg["probes"]["K"])
    probe_batch = int(cfg["probes"].get("probe_batch", 8))
    base_batch = int(cfg["base_trajectories"].get("batch_size", 8))
    seed = int(cfg.get("seed", 42))

    N_total = int(cfg["base_trajectories"]["N"])
    n_local = N_total // world_size
    if rank < N_total % world_size:
        n_local += 1

    if rank == 0:
        print(f"\nGenerating {N_total} base trajectories "
              f"({num_solver_steps}-step Euler, batch={base_batch}) ...")

    cache_local, labels_local = sample_base_trajectories(
        model=model,
        n_local=n_local,
        num_classes=int(cfg["model"].get("num_classes", 1000)),
        num_solver_steps=num_solver_steps,
        device=device,
        seed=seed,
        rank=rank,
        world_size=world_size,
        batch_size=base_batch,
    )

    # Sanity check: cache at t=1 vs directly integrated endpoint (rank 0 only).
    # Use the same batch size as the caching pass to avoid tiny batch-order
    # rounding differences in the transformer matmuls.
    if rank == 0:
        nb = min(base_batch, cache_local[0].shape[0])
        z0 = cache_local[0][:nb].to(device)
        y0 = labels_local[:nb]
        z1_direct = integrate_partial(model, z0, y0, 0, num_solver_steps)
        z1_cached = cache_local[num_solver_steps][:nb].to(device)
        diff = (z1_direct - z1_cached).abs().max().item()
        print(f"  Cache sanity: max |z1_direct - z1_cached| = {diff:.3e}")

    barrier(world_size)

    # ── dispatch ─────────────────────────────────────────────────────────────
    if mode == "stability":
        run_stability(cfg, model, cache_local, labels_local, device, rank,
                      world_size, exp_dir, seed, num_solver_steps,
                      probe_batch, K)
    elif mode == "wavg":
        run_wavg(cfg, model, cache_local, labels_local, device, rank,
                 world_size, exp_dir, seed, num_solver_steps, probe_batch, K)

    if rank == 0:
        print("\nDone.")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
