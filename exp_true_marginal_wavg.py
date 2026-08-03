"""
exp_true_marginal_wavg.py -- Analytic-marginal path-weighted amplification
                             w_avg(T, t) on the empirical ImageNet latent
                             distribution.

Revision v2 (2026-08-03): fixes reported by the revision doc
tmp/true_marginal_wavg_required_revisions.html:

  * Estimate w_avg(T, t) on [0, T] for a *fixed* T < 1 (default T = 0.99),
    with step h = T / S. No integration into the endpoint singularity.
  * Numerically stable responsibility logits (shift-free formulation).
  * Centering-corrected covariance action  C_alpha a  including the mu * sum
    correction term (zero in exact arithmetic; stabilizing in FP32).
  * FP64 reference mode (config: estimator.precision).
  * Autograd VJP validation of the local J^T operator.
  * Trace-averaged forward/backward validation (not per-probe norm compare).
  * Probe-level uncertainty preserved and exported.
  * Continuous-time alignment of solver-convergence comparisons.
  * Marginal-moment validation of the integrated flow.
  * Explicit latent-loader mode  (precomputed_point | posterior_mean |
    fixed_posterior_sample), explicit latent_scale, no hidden 0.18215.
  * Outlier report (any class or trajectory contributing > 20% flagged).
  * Log-scale plots and quantile-annotated histograms.

Notation
--------
Given a class c with target latents  {y_i^(c)}_{i=1..M_c},  prior
z_0 ~ N(0, I),  and the linear interpolant  z_t = (1 - t) z_0 + t y_i,
the induced conditional marginal velocity is

    alpha_i(z, t | c) = softmax_i(  (t / (1-t)^2) <z, y_i>
                                  - (t^2 / (2 (1-t)^2)) ||y_i||^2 )
    mu_c(z, t)        = sum_i alpha_i y_i
    v_c(z, t)         = ( mu_c(z, t) - z ) / (1 - t)

The instantaneous Jacobian is

    J_v(z, t) = -1/(1-t) I + t / (1-t)^3 * C_alpha,
    C_alpha = sum_i alpha_i (y_i - mu)(y_i - mu)^T.

We integrate S explicit-Euler steps  z_{n+1} = z_n + h v(z_n, t_n)
with  h = T / S,  t_n = n h,  n = 0 .. S-1, terminating at t_S = T < 1.
The discrete Euler-step Jacobian is  G_n = I + h J_v(z_n, t_n).

The estimator uses K unit-norm Rademacher probes anchored at t = T:

    w_avg(T, t_n) = E [ || Phi(T, t_n)^T q ||^2 ],
    Phi(T, t_n) = G_{S-1} G_{S-2} ... G_n.

A backward adjoint sweep from n = S down to n = 0 records ||a_n||^2 at
every reporting index.  Terminal identity  w(T, T) = 1  arises from the
initialization a_S = q, not from a special-case override.

Distribution
------------
Classes are round-robin-sharded across ranks. No cross-rank comm during
compute; only aggregation on rank 0 at the end.

Usage
-----
Multi-GPU (recommended):
    torchrun --standalone --nproc_per_node=8 exp_true_marginal_wavg.py \
        --config configs/true_marginal_wavg_imagenet.yaml \
        --output experiment/true_marginal_wavg_imagenet_T099_S256

Single-GPU debug:
    python exp_true_marginal_wavg.py --config configs/... --device cuda:0
"""

import argparse
import json
import math
import os
import shutil
import time
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import yaml


# ============================================================
# Constants (matches encode_dataset.py / train.py: 256x256 -> 4x32x32 latents)
# ============================================================

DEFAULT_LATENT_C = 4
DEFAULT_LATENT_H = 32
DEFAULT_LATENT_W = 32
DEFAULT_LATENT_D = DEFAULT_LATENT_C * DEFAULT_LATENT_H * DEFAULT_LATENT_W   # 4096

# NumPy 2.0 renamed `trapz` -> `trapezoid`.
_np_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# ============================================================
# Distributed setup
# ============================================================

def setup_dist(default_device: str):
    """Initialize NCCL when torchrun sets RANK/WORLD_SIZE; else single-GPU."""
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


# ============================================================
# Config
# ============================================================

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ============================================================
# Logger
# ============================================================

class JsonlLogger:
    """Per-rank JSONL writer; rank 0 also mirrors messages to stdout."""

    def __init__(self, path: str, rank: int, world_size: int, echo: bool = True):
        self.path = path
        self.rank = rank
        self.world_size = world_size
        self.echo = echo and (rank == 0)
        self._f = open(path, "a", buffering=1)
        self._t0 = time.time()

    def log(self, record: dict, stdout_msg: str = None):
        record = dict(record)
        record.setdefault("rank", self.rank)
        record.setdefault("elapsed_sec", round(time.time() - self._t0, 3))
        record.setdefault("timestamp", datetime.utcnow().isoformat(timespec="seconds"))
        self._f.write(json.dumps(record) + "\n")
        if self.echo and stdout_msg:
            print(stdout_msg, flush=True)

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ============================================================
# Packed-latent loader (matches train.py's PackedLatentImageFolder format)
# ============================================================

def list_packed_classes(latent_root: str) -> list:
    """Sorted list of class names (WordNet IDs) with a .npy file present."""
    if not os.path.isdir(latent_root):
        raise FileNotFoundError(f"latent_root does not exist: {latent_root}")
    files = sorted(f for f in os.listdir(latent_root) if f.endswith(".npy"))
    if not files:
        raise RuntimeError(f"No .npy files found under {latent_root}")
    return [os.path.splitext(f)[0] for f in files]


def load_class_latents(
    latent_root: str,
    class_name: str,
    device: torch.device,
    seed: int,
    mode: str,
    scale: float,
    flip: str = "original",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Load all pre-encoded latents of one ImageNet class into a (M_c, D) tensor.

    The .npy file has shape (M_c, 2, 2, C, H, W) fp32:
        axis 1 (orientation) : 0 = original, 1 = horizontally flipped
        axis 2 (parameter)   : 0 = mean,     1 = std

    Args:
        mode : latent semantic
               * "precomputed_point"      -> use posterior mean, no eps, no scale
                                             (treat each cached mean as a fixed point)
               * "posterior_mean"         -> use posterior mean, no eps, apply scale
                                             (mean * scale)
               * "fixed_posterior_sample" -> (mean + std * eps) * scale
                                             with eps deterministically seeded
        scale : multiplicative factor applied to y_i (usually 0.18215 to match
                training input, or 1.0 for raw-latent runs).

    Returns:
        Y : (M_c, D) tensor of the requested `dtype`, on `device`.
    """
    ori_idx = 0 if flip == "original" else 1
    path = os.path.join(latent_root, class_name + ".npy")
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 6 or arr.shape[1] < ori_idx + 1 or arr.shape[2] != 2:
        raise RuntimeError(f"Unexpected packed-latent shape at {path}: {arr.shape}")

    mean_np = np.array(arr[:, ori_idx, 0], dtype=np.float32, copy=True)   # (M, C, H, W)
    if mode == "fixed_posterior_sample":
        std_np = np.array(arr[:, ori_idx, 1], dtype=np.float32, copy=True)
    del arr

    mean_t = torch.from_numpy(mean_np).to(device=device, dtype=dtype)

    if mode == "precomputed_point":
        y = mean_t                                          # ignore scale entirely
    elif mode == "posterior_mean":
        y = mean_t * float(scale)
    elif mode == "fixed_posterior_sample":
        std_t = torch.from_numpy(std_np).to(device=device, dtype=dtype)
        gen = torch.Generator(device=device).manual_seed(int(seed))
        eps = torch.randn(mean_t.shape, device=device,
                          dtype=dtype, generator=gen)
        y = (mean_t + std_t * eps) * float(scale)
    else:
        raise ValueError(f"Unknown latent_mode: {mode!r}. "
                         "Choose from: precomputed_point, posterior_mean, "
                         "fixed_posterior_sample.")

    return y.reshape(y.shape[0], -1).contiguous()           # (M, D)


# ============================================================
# Analytic velocity + Jacobian-transpose action
# ============================================================
#
# Stable logit form (revision doc, section 3):
#   ell_i = (t / (1-t)^2) <z, y_i>  -  (t^2 / (2 (1-t)^2)) ||y_i||^2
# The ||z||^2 contribution is shared by all i and drops out under softmax.

@torch.no_grad()
def responsibilities(
    Z: torch.Tensor,        # (N, D)
    Y: torch.Tensor,        # (M, D)
    y_sq: torch.Tensor,     # (1, M)  precomputed ||y_i||^2
    t: float,
) -> tuple:
    """Numerically stable responsibilities in the working precision of Z, Y."""
    s = 1.0 - t
    inv_s2 = 1.0 / (s * s)
    zy = Z @ Y.T                                                # (N, M)
    logits = (t * inv_s2) * zy - (0.5 * t * t * inv_s2) * y_sq  # (N, M)
    alpha = torch.softmax(logits, dim=1)
    mu = alpha @ Y                                              # (N, D)
    return alpha, mu


@torch.no_grad()
def velocity(Z, Y, y_sq, t):
    alpha, mu = responsibilities(Z, Y, y_sq, t)
    v = (mu - Z) / (1.0 - t)
    return v, alpha, mu


@torch.no_grad()
def apply_JT_batched(
    A: torch.Tensor,        # (K, N, D)
    Y: torch.Tensor,        # (M, D)
    t: float,
    alpha: torch.Tensor,    # (N, M)
    mu: torch.Tensor,       # (N, D)
) -> torch.Tensor:
    """Apply  J_v(z, t)^T  to a batch of adjoints, with centering correction.

    J_v = -1/(1-t) I + t / (1-t)^3 * C_alpha
    with  C_alpha = sum_i alpha_i (y_i - mu)(y_i - mu)^T.

    Numerically stable form (revision doc section 3):
        s_i = <y_i, a> - <mu, a>
        r_i = alpha_i * s_i
        C_alpha a = Y^T r  -  mu * (sum_i r_i)
    In exact arithmetic sum_i r_i = 0 (since sum_i alpha_i (y_i - mu) = 0),
    but subtracting the roundoff of  mu * sum r  measurably improves FP32
    accuracy when one alpha_i is close to 1.
    """
    one_minus_t = 1.0 - t
    c1 = -1.0 / one_minus_t
    c2 = t / (one_minus_t ** 3)

    K, N, D = A.shape
    M = Y.shape[0]

    A_flat = A.reshape(K * N, D)                                # (K*N, D)

    mu_dot_a = (A * mu.unsqueeze(0)).sum(dim=-1)                # (K, N)
    mu_dot_a_flat = mu_dot_a.reshape(K * N, 1)                  # (K*N, 1)

    Ya = A_flat @ Y.T                                           # (K*N, M)
    s = Ya - mu_dot_a_flat                                      # (K*N, M)

    alpha_flat = alpha.unsqueeze(0).expand(K, N, M).reshape(K * N, M)
    r = alpha_flat * s                                          # (K*N, M)

    Ca = r @ Y                                                  # (K*N, D)

    # Centering correction: subtract  mu * sum_i r_i  (zero in exact arith.)
    r_sum = r.sum(dim=1, keepdim=True)                          # (K*N, 1)
    mu_expanded = mu.unsqueeze(0).expand(K, N, D).reshape(K * N, D)
    Ca = Ca - r_sum * mu_expanded

    Ca = Ca.reshape(K, N, D)

    return c1 * A + c2 * Ca


# ============================================================
# Differentiable velocity (used only by autograd-VJP validation)
# ============================================================

def _velocity_diff(z: torch.Tensor, Y: torch.Tensor, t: float) -> torch.Tensor:
    """Differentiable analytic velocity for a single trajectory z (D,) -> v (D,).
    Small helper for autograd.grad-based VJP checks; do NOT call in the hot loop.
    """
    s = 1.0 - t
    inv_s2 = 1.0 / (s * s)
    zy = Y @ z                                                  # (M,)
    y_sq = (Y * Y).sum(dim=1)                                   # (M,)
    logits = (t * inv_s2) * zy - (0.5 * t * t * inv_s2) * y_sq
    alpha = torch.softmax(logits, dim=0)
    mu = alpha @ Y
    return (mu - z) / s


# ============================================================
# Forward flow (analytic velocity) + adjoint sweep
# ============================================================

def run_class(
    Y: torch.Tensor,             # (M_c, D)
    class_seed: int,
    N: int, K: int, S: int,
    T_end: float,
    report_indices: list,
    device: torch.device,
    logger: JsonlLogger,
    rank: int,
    class_local_idx: int, class_local_total: int,
    class_global_idx: int, class_global_total: int,
    progress_interval: int = 8,
    resp_stats_every: int = 32,
    latency_probe_interval: int = 32,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Forward Euler flow on [0, T_end] with S steps + backward adjoint sweep.

    Returns a dict with:
        * class_mean         : (T,)  E over probes and trajectories
        * class_traj_mean    : (N, T)  mean over probes per trajectory
        * class_traj_probe   : (N, K, T)  full per-(traj, probe) tensor
                               (kept in float32 to bound memory)
        * probe_var_by_traj  : (N, T)  Var over K probes per trajectory
        * probe_var_mean     : (T,)   mean over trajectories of the above
    """
    M, D = Y.shape
    h = T_end / S
    y_sq = (Y * Y).sum(dim=1, keepdim=True).T                   # (1, M)

    with torch.no_grad():
        Z_cache = torch.empty((S + 1, N, D), dtype=dtype, device=device)
        gen = torch.Generator(device=device).manual_seed(int(class_seed))
        Z_cache[0] = torch.randn((N, D), device=device, dtype=dtype, generator=gen)

    # ---- forward Euler flow ----------------------------------------------
    v_latency = []
    resp_entropy_samples = []
    resp_neff_samples = []
    resp_sumcheck_max_dev = 0.0
    fwd_t0 = time.time()

    for n in range(S):
        t = n * h
        if latency_probe_interval and (n % latency_probe_interval == 0):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_call0 = time.time()

        with torch.no_grad():
            v, alpha, mu = velocity(Z_cache[n], Y, y_sq, t)

        if latency_probe_interval and (n % latency_probe_interval == 0):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            v_latency.append((time.time() - t_call0) * 1000.0)

        if resp_stats_every and (n % resp_stats_every == 0):
            a_sums = alpha.sum(dim=1)
            resp_sumcheck_max_dev = max(
                resp_sumcheck_max_dev,
                float((a_sums - 1.0).abs().max().item()),
            )
            H_a = -(alpha * (alpha.clamp_min(1e-30)).log()).sum(dim=1)
            neff = 1.0 / (alpha * alpha).sum(dim=1).clamp_min(1e-30)
            resp_entropy_samples.append(float(H_a.mean().item()))
            resp_neff_samples.append(float(neff.mean().item()))

        with torch.no_grad():
            Z_cache[n + 1] = Z_cache[n] + h * v

        if not torch.isfinite(Z_cache[n + 1]).all():
            raise RuntimeError(
                f"[rank {rank}] non-finite state at step {n} "
                f"(class local {class_local_idx}/{class_local_total})"
            )

        if rank == 0 and progress_interval and \
                ((n + 1) % progress_interval == 0 or n == S - 1):
            elapsed = time.time() - fwd_t0
            eta = elapsed * (S - n - 1) / max(1, n + 1)
            v_ms = np.median(v_latency) if v_latency else float("nan")
            msg = (f"[rank {rank}] phase=forward "
                   f"class_global={class_global_idx:03d}/{class_global_total:02d} "
                   f"class_local={class_local_idx:03d}/{class_local_total:02d} "
                   f"M_c={M} sample=N={N} step={n + 1:03d}/{S} "
                   f"t={t:.6f} v_batch_ms={v_ms:.2f} "
                   f"elapsed={fmt_time(elapsed)} eta_class={fmt_time(eta)}")
            logger.log({
                "phase": "forward", "class_global": class_global_idx,
                "class_local": class_local_idx, "step": n + 1, "S": S,
                "t": t, "T_end": T_end, "M_c": M, "v_batch_ms": v_ms,
            }, stdout_msg=msg)

    fwd_time = time.time() - fwd_t0

    # ---- backward adjoint sweep ------------------------------------------
    with torch.no_grad():
        q_gen = torch.Generator(device=device).manual_seed(int(class_seed) + 999983)
        r = torch.empty((K, N, D), device=device, dtype=dtype)
        r.bernoulli_(0.5, generator=q_gen).mul_(2.0).sub_(1.0)
        r.div_(math.sqrt(D))                                     # ||q|| = 1
        A = r

    idx_to_pos = {s: i for i, s in enumerate(report_indices)}
    T_rep = len(report_indices)
    # (T, N, K) in fp64 for reduction; per-probe raw kept in fp32
    w_traj_probe = torch.zeros((T_rep, N, K), dtype=torch.float64, device=device)

    if S in idx_to_pos:
        # Terminal norm at t = T is  ||q||^2, measured from the initialized A
        # (not hard-coded to 1; robust to any future non-unit probe scaling).
        sq0 = (A.to(torch.float64) * A.to(torch.float64)).sum(dim=-1)   # (K, N)
        w_traj_probe[idx_to_pos[S]] = sq0.T                              # (N, K)

    jtv_latency = []
    adj_t0 = time.time()

    for n in range(S - 1, -1, -1):
        t = n * h
        with torch.no_grad():
            alpha, mu = responsibilities(Z_cache[n], Y, y_sq, t)

        if latency_probe_interval and (n % latency_probe_interval == 0):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_call0 = time.time()

        with torch.no_grad():
            JT_A = apply_JT_batched(A, Y, t, alpha, mu)

        if latency_probe_interval and (n % latency_probe_interval == 0):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            jtv_latency.append((time.time() - t_call0) * 1000.0)

        with torch.no_grad():
            A = A + h * JT_A

        if not torch.isfinite(A).all():
            raise RuntimeError(
                f"[rank {rank}] non-finite adjoint at step {n} "
                f"(class local {class_local_idx}/{class_local_total})"
            )

        if n in idx_to_pos:
            sq = (A.to(torch.float64) * A.to(torch.float64)).sum(dim=-1)   # (K, N)
            w_traj_probe[idx_to_pos[n]] = sq.T                              # (N, K)

        if rank == 0 and progress_interval and \
                ((S - n) % progress_interval == 0 or n == 0):
            elapsed = time.time() - adj_t0
            done = S - n
            eta = elapsed * (S - done) / max(1, done)
            jtv_ms = np.median(jtv_latency) if jtv_latency else float("nan")
            partial = float(A.reshape(K * N, D).to(torch.float64).pow(2).sum().item()
                            / (K * N))
            msg = (f"[rank {rank}] phase=adjoint "
                   f"class_global={class_global_idx:03d}/{class_global_total:02d} "
                   f"class_local={class_local_idx:03d}/{class_local_total:02d} "
                   f"sample=N={N} probe=K={K} backward_step={done:03d}/{S} "
                   f"t={t:.6f} jtv_batch_ms={jtv_ms:.2f} "
                   f"w_partial={partial:.4f} elapsed={fmt_time(elapsed)} "
                   f"eta_class={fmt_time(eta)}")
            logger.log({
                "phase": "adjoint", "class_global": class_global_idx,
                "class_local": class_local_idx, "backward_step": done,
                "S": S, "t": t, "T_end": T_end,
                "jtv_batch_ms": jtv_ms, "w_partial": partial,
            }, stdout_msg=msg)

    adj_time = time.time() - adj_t0

    # Aggregations
    w_np = w_traj_probe.cpu().numpy()                                   # (T, N, K)
    class_mean = w_np.mean(axis=(1, 2))                                 # (T,)
    class_traj_mean = w_np.mean(axis=2).T                               # (N, T)
    class_traj_probe = np.transpose(w_np, (1, 2, 0)).astype(np.float32) # (N, K, T)
    probe_var_by_traj = w_np.var(axis=2, ddof=1).T                      # (N, T)
    probe_var_mean = probe_var_by_traj.mean(axis=0)                     # (T,)

    peak_mem_gib = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)) \
        if device.type == "cuda" else 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    return {
        "class_mean":          class_mean.astype(np.float64),
        "class_traj_mean":     class_traj_mean.astype(np.float64),
        "class_traj_probe":    class_traj_probe,               # (N, K, T) fp32
        "probe_var_by_traj":   probe_var_by_traj.astype(np.float64),
        "probe_var_mean":      probe_var_mean.astype(np.float64),
        "M_c": int(M), "N": int(N), "K": int(K), "S": int(S),
        "T_end": float(T_end),
        "report_indices": list(report_indices),
        "actual_times": [float(s_i * (T_end / S)) for s_i in report_indices],
        "fwd_time_sec": float(fwd_time),
        "adj_time_sec": float(adj_time),
        "mean_v_batch_ms":  float(np.mean(v_latency)) if v_latency else float("nan"),
        "mean_jtv_batch_ms":float(np.mean(jtv_latency)) if jtv_latency else float("nan"),
        "p50_v_batch_ms":   float(np.median(v_latency)) if v_latency else float("nan"),
        "p90_v_batch_ms":   float(np.quantile(v_latency, 0.9)) if v_latency else float("nan"),
        "p50_jtv_batch_ms": float(np.median(jtv_latency)) if jtv_latency else float("nan"),
        "p90_jtv_batch_ms": float(np.quantile(jtv_latency, 0.9)) if jtv_latency else float("nan"),
        "peak_gpu_mem_gib": peak_mem_gib,
        "resp_entropy_mean":    float(np.mean(resp_entropy_samples)) if resp_entropy_samples else float("nan"),
        "resp_neff_mean":       float(np.mean(resp_neff_samples)) if resp_neff_samples else float("nan"),
        "resp_sumcheck_max_dev":float(resp_sumcheck_max_dev),
        "dtype": str(dtype),
    }


# ============================================================
# Validation: autograd VJP, trace-averaged FD, marginal moments
# ============================================================

def autograd_vjp_validation(Y: torch.Tensor, t: float,
                            N_val: int, K_val: int,
                            device: torch.device, seed: int,
                            dtype: torch.dtype) -> dict:
    """Compare analytic  J_v^T a  with autograd  d/dz [a . v(z, t)].

    Doc requirement: rel error < 1e-4 in FP32,  < 1e-8 in FP64.
    """
    M, D = Y.shape
    y_sq = (Y * Y).sum(dim=1, keepdim=True).T
    gen = torch.Generator(device=device).manual_seed(int(seed))

    Z = torch.randn((N_val, D), device=device, dtype=dtype, generator=gen)
    A = torch.randn((K_val, N_val, D), device=device, dtype=dtype, generator=gen)

    # Analytic path.
    with torch.no_grad():
        alpha, mu = responsibilities(Z, Y, y_sq, t)
        JT_analytic = apply_JT_batched(A, Y, t, alpha, mu)              # (K, N, D)

    # Autograd path: for each trajectory, compute grad_z (a . v(z, t)).
    JT_autograd = torch.zeros_like(JT_analytic)
    for n in range(N_val):
        for k in range(K_val):
            z = Z[n].detach().clone().requires_grad_(True)
            v = _velocity_diff(z, Y, t)
            scalar = (A[k, n] * v).sum()
            g = torch.autograd.grad(scalar, z, create_graph=False)[0]
            JT_autograd[k, n] = g

    diff = (JT_analytic - JT_autograd).to(torch.float64)
    ref = JT_autograd.to(torch.float64)
    rel_err = diff.norm() / (ref.norm() + 1e-30)

    return {
        "t": float(t), "N_val": int(N_val), "K_val": int(K_val),
        "dtype": str(dtype),
        "rel_error_analytic_vs_autograd": float(rel_err.item()),
        "analytic_norm": float(JT_analytic.to(torch.float64).norm().item()),
        "autograd_norm": float(JT_autograd.to(torch.float64).norm().item()),
    }


def cov_action_reference_fp64(Y: torch.Tensor, t: float, seed: int) -> dict:
    """Explicit  C_alpha a  in float64 vs the batched implementation.

    Explicit  C_alpha  is (D, D) and expensive; run on a tiny D to make it
    tractable.  Called only from the unit-test / diagnostic pathway.
    """
    device = Y.device
    y_sq = (Y * Y).sum(dim=1, keepdim=True).T

    gen = torch.Generator(device=device).manual_seed(int(seed))
    z = torch.randn((1, Y.shape[1]), device=device,
                    dtype=torch.float64, generator=gen)
    a = torch.randn((1, 1, Y.shape[1]), device=device,
                    dtype=torch.float64, generator=gen)

    alpha, mu = responsibilities(z, Y, y_sq, t)                # (1, M), (1, D)

    # Explicit C_alpha (D, D).
    Yc = Y - mu                                                # (M, D)
    Ca_explicit = (alpha[0].unsqueeze(1) * Yc).T @ Yc          # (D, D)
    ref = Ca_explicit @ a[0, 0]                                # (D,)

    # Batched path.
    JT = apply_JT_batched(a, Y, t, alpha, mu)                  # (1, 1, D)
    # Reconstruct C_alpha a from JT: JT = c1 a + c2 (C_alpha a).
    s = 1.0 - t
    c1 = -1.0 / s
    c2 = t / (s ** 3)
    batched = (JT[0, 0] - c1 * a[0, 0]) / c2

    diff = (batched - ref).norm()
    rel_err = diff / (ref.norm() + 1e-30)
    return {
        "t": float(t),
        "rel_error_batched_vs_explicit": float(rel_err.item()),
        "explicit_norm": float(ref.norm().item()),
        "batched_norm":  float(batched.norm().item()),
    }


def trace_averaged_fd_validation(
    Y: torch.Tensor, Y_fp64: torch.Tensor,
    class_seed: int, N_val: int, K_val: int,
    S: int, T_end: float, solver_index: int,
    delta_list: list, device: torch.device, dtype: torch.dtype,
) -> dict:
    """Trace-averaged forward FD vs backward-analytic  w_avg  agreement.

    Revision doc, section 3, "Full-flow validation":
        w_back = mean_k || Phi^T q_k ||^2
        w_fd   = mean_k || (F(z + delta g_k) - F(z - delta g_k)) / (2 delta) ||^2
    Report  |w_fd - w_back| / (w_back + eps).
    """
    M, D = Y.shape
    y_sq = (Y * Y).sum(dim=1, keepdim=True).T
    h = T_end / S

    # --- Common trajectories and probes -----------------------------------
    gen = torch.Generator(device=device).manual_seed(int(class_seed))
    Z0 = torch.randn((N_val, D), device=device, dtype=dtype, generator=gen)

    Z = Z0.clone()
    Z_states = [Z]
    with torch.no_grad():
        for n in range(S):
            t = n * h
            v, _, _ = velocity(Z, Y, y_sq, t)
            Z = Z + h * v
            Z_states.append(Z)

    Z_mid = Z_states[solver_index]                             # (N, D)

    q_gen = torch.Generator(device=device).manual_seed(int(class_seed) + 314159)
    q = torch.empty((K_val, N_val, D), device=device, dtype=dtype)
    q.bernoulli_(0.5, generator=q_gen).mul_(2.0).sub_(1.0).div_(math.sqrt(D))

    # --- backward analytic  w_back  -------------------------------------
    A = q.clone()
    with torch.no_grad():
        for n in range(S - 1, solver_index - 1, -1):
            t = n * h
            alpha, mu = responsibilities(Z_states[n], Y, y_sq, t)
            JT_A = apply_JT_batched(A, Y, t, alpha, mu)
            A = A + h * JT_A
    w_back_per_probe = A.to(torch.float64).pow(2).sum(dim=-1)   # (K, N)
    w_back = float(w_back_per_probe.mean().item())

    # --- forward FD  w_fd  ----------------------------------------------
    def _integrate_from(Z_start, s_start):
        Zc = Z_start
        with torch.no_grad():
            for n in range(s_start, S):
                t = n * h
                v, _, _ = velocity(Zc, Y, y_sq, t)
                Zc = Zc + h * v
        return Zc

    z_rms = Z_mid.pow(2).mean(dim=1, keepdim=True).sqrt()       # (N, 1)

    fd_results = {}
    for delta_eta in delta_list:
        # delta chosen so that per-coord RMS matches the state's RMS scale.
        delta = float(delta_eta) * z_rms.unsqueeze(0) * math.sqrt(D)   # (1, N, 1)
        Zp = (Z_mid.unsqueeze(0) + delta * q).reshape(-1, D)
        Zm = (Z_mid.unsqueeze(0) - delta * q).reshape(-1, D)
        Fp = _integrate_from(Zp, solver_index).reshape(K_val, N_val, D)
        Fm = _integrate_from(Zm, solver_index).reshape(K_val, N_val, D)
        tang = (Fp - Fm) / (2.0 * delta.squeeze(-1).unsqueeze(-1))
        w_fd_per_probe = tang.to(torch.float64).pow(2).sum(dim=-1)     # (K, N)
        w_fd = float(w_fd_per_probe.mean().item())
        rel_err = abs(w_fd - w_back) / (w_back + 1e-12)
        fd_results[float(delta_eta)] = {
            "w_fd": w_fd, "w_back": w_back, "rel_err": rel_err,
        }

    # Recommended stable-plateau delta = median of the tested values.
    return {
        "solver_index": int(solver_index),
        "t": float(solver_index * h),
        "T_end": float(T_end), "S": int(S),
        "N_val": int(N_val), "K_val": int(K_val),
        "w_back": w_back,
        "fd_by_delta": fd_results,
    }


def marginal_moment_validation(
    Y: torch.Tensor, class_seed: int,
    N_val: int, S: int, T_end: float,
    report_indices: list, device: torch.device, dtype: torch.dtype,
) -> dict:
    """Compare integrated sample moments of z_t vs the target marginal.

    Under the ideal marginal path,
        E[z_t | c]     = t * y_bar_c
        Cov(z_t | c)   = (1-t)^2 I + t^2 Cov(y | c).
    We compare integrated  z_n  moments to these predictions at each report index.
    """
    M, D = Y.shape
    y_sq = (Y * Y).sum(dim=1, keepdim=True).T
    h = T_end / S

    y_bar = Y.mean(dim=0)                                                # (D,)
    trace_cov_y = float(((Y - y_bar).to(torch.float64) ** 2).sum().item() / max(1, M))
    # (Note: Cov(y) is (D, D); trace = mean over i of ||y_i - y_bar||^2.)

    gen = torch.Generator(device=device).manual_seed(int(class_seed))
    Z = torch.randn((N_val, D), device=device, dtype=dtype, generator=gen)

    idx_to_pos = {s: i for i, s in enumerate(report_indices)}
    T_rep = len(report_indices)
    mean_rel_err = np.zeros(T_rep)
    tr_cov_rel_err = np.zeros(T_rep)

    def _record(pos, Zc, t):
        emp_mean = Zc.mean(dim=0)                                       # (D,)
        pred_mean = t * y_bar
        num = (emp_mean - pred_mean).to(torch.float64).norm().item()
        den = pred_mean.to(torch.float64).norm().item() + 1e-30
        mean_rel_err[pos] = num / den

        Zc_centered = Zc - emp_mean
        tr_emp = float((Zc_centered.to(torch.float64) ** 2).sum().item() / max(1, N_val))
        # Predicted trace: (1-t)^2 D  +  t^2 * trace(Cov(y|c))
        tr_pred = (1.0 - t) ** 2 * D + (t ** 2) * trace_cov_y
        tr_cov_rel_err[pos] = abs(tr_emp - tr_pred) / (tr_pred + 1e-30)

    with torch.no_grad():
        if 0 in idx_to_pos:
            _record(idx_to_pos[0], Z, 0.0)
        for n in range(S):
            t = n * h
            v, _, _ = velocity(Z, Y, y_sq, t)
            Z = Z + h * v
            if (n + 1) in idx_to_pos:
                _record(idx_to_pos[n + 1], Z, (n + 1) * h)

    return {
        "mean_rel_err": mean_rel_err,
        "cov_trace_rel_err": tr_cov_rel_err,
        "actual_times": [float(s_i * h) for s_i in report_indices],
    }


# ============================================================
# Aggregation, plotting, output writing (rank 0 only)
# ============================================================

def bootstrap_ci_axis0(values: np.ndarray, n_boot: int, seed: int,
                       alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    if n < 2:
        return values.mean(axis=0), values.mean(axis=0)
    means = np.empty((n_boot,) + values.shape[1:], dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = values[idx].mean(axis=0)
    lo = np.quantile(means, alpha / 2, axis=0)
    hi = np.quantile(means, 1 - alpha / 2, axis=0)
    return lo, hi


def _lazy_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def write_outputs(
    out_dir: str,
    t_grid: np.ndarray,
    T_end: float,
    selected_class_ids: list,
    per_class_mean: np.ndarray,           # (C, T)
    per_class_traj_mean: np.ndarray,      # (C, N, T)
    per_class_probe_var_mean: np.ndarray, # (C, T)
    per_class_meta: list,
    bootstrap_replicates: int,
    seed: int,
) -> None:
    """Write all tables, plots, master npz, and the outlier report."""
    os.makedirs(out_dir, exist_ok=True)
    C, T_rep = per_class_mean.shape
    plt = _lazy_plt()
    from matplotlib.colors import LogNorm

    # -------- global mean + CI ---------------------------------------------
    global_mean = per_class_mean.mean(axis=0)
    global_median = np.median(per_class_mean, axis=0)
    ci_lo, ci_hi = bootstrap_ci_axis0(per_class_mean, bootstrap_replicates, seed)
    q10 = np.quantile(per_class_mean, 0.10, axis=0)
    q90 = np.quantile(per_class_mean, 0.90, axis=0)

    # -------- wavg_global.csv ---------------------------------------------
    with open(os.path.join(out_dir, "wavg_global.csv"), "w") as f:
        f.write("t,mean,median,q10,q90,ci95_low,ci95_high\n")
        for j in range(T_rep):
            f.write(f"{t_grid[j]:.10f},{global_mean[j]:.10e},"
                    f"{global_median[j]:.10e},{q10[j]:.10e},{q90[j]:.10e},"
                    f"{ci_lo[j]:.10e},{ci_hi[j]:.10e}\n")

    # -------- wavg_per_class.csv ------------------------------------------
    with open(os.path.join(out_dir, "wavg_per_class.csv"), "w") as f:
        f.write("class_id,class_name,"
                + ",".join(f"t={t_grid[j]:.6f}" for j in range(T_rep)) + "\n")
        for c in range(C):
            row = ",".join(f"{per_class_mean[c, j]:.10e}" for j in range(T_rep))
            f.write(f"{c},{selected_class_ids[c]},{row}\n")

    # -------- Global curve (log-y) with quantile band ----------------------
    # Curve ends at T_end (not extended to t=1).
    for scale, fname in [("linear", "wavg_global_linear.png"),
                         ("log",    "wavg_global_logy.png")]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(t_grid, global_mean, marker="o", linewidth=2, color="steelblue",
                label="global mean")
        ax.plot(t_grid, global_median, marker="s", linewidth=1.2,
                color="darkorange", label="global median")
        ax.fill_between(t_grid, q10, q90, alpha=0.20, color="steelblue",
                        label="10-90% class range")
        ax.fill_between(t_grid, ci_lo, ci_hi, alpha=0.30, color="orchid",
                        label="95% class-bootstrap CI")
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="w=1")
        if scale == "log":
            ax.set_yscale("log")
        ax.set_xlabel(f"t   (T_end = {T_end:g})")
        ax.set_ylabel("w_avg(T, t)")
        ax.set_title(f"Analytic-marginal w_avg(T={T_end:g}, t)   ({scale}-y)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=140)
        plt.close()

    # -------- Log-normalized heatmap --------------------------------------
    heat = np.clip(per_class_mean, 1e-30, None)
    A_c = _np_trapezoid(per_class_mean, t_grid, axis=1)
    order = np.argsort(A_c)

    for tag, mat in [("original", heat), ("sorted", heat[order])]:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(mat, aspect="auto", origin="lower",
                       extent=[t_grid[0], t_grid[-1], 0, C],
                       cmap="viridis",
                       norm=LogNorm(vmin=max(1e-6, mat.min()), vmax=mat.max()))
        ax.set_xlabel("t")
        ax.set_ylabel("class rank" + (" (sorted)" if tag == "sorted" else ""))
        ax.set_title(f"w_avg(T={T_end:g}, t) per class  ({tag}, log color)")
        plt.colorbar(im, ax=ax, label="w_avg(t)  (log)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"wavg_class_time_heatmap_{tag}.png"),
                    dpi=140)
        plt.close()

    # linear versions (secondary diagnostic)
    for tag, mat in [("original", heat), ("sorted", heat[order])]:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(mat, aspect="auto", origin="lower",
                       extent=[t_grid[0], t_grid[-1], 0, C], cmap="viridis")
        ax.set_xlabel("t")
        ax.set_ylabel("class rank" + (" (sorted)" if tag == "sorted" else ""))
        ax.set_title(f"w_avg(T={T_end:g}, t) per class  ({tag}, linear color)")
        plt.colorbar(im, ax=ax, label="w_avg(t)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,
                                 f"wavg_class_time_heatmap_{tag}_linear.png"),
                    dpi=140)
        plt.close()

    # -------- All-class overlays -------------------------------------------
    for scale, fname in [("linear", "wavg_all_classes_linear.png"),
                         ("log",    "wavg_all_classes_logy.png")]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for c in range(C):
            ax.plot(t_grid, per_class_mean[c], color="steelblue",
                    linewidth=0.5, alpha=0.35)
        ax.plot(t_grid, global_mean,   color="black",     linewidth=2.2,
                label="global mean")
        ax.plot(t_grid, global_median, color="darkorange", linewidth=1.6,
                label="global median")
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel(f"t   (T_end = {T_end:g})")
        ax.set_ylabel("w_avg(t)")
        if scale == "log":
            ax.set_yscale("log")
        ax.set_title(f"w_avg(t) for {C} classes")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=140)
        plt.close()

    # -------- Log-x histograms + ECDFs at selected times -------------------
    rep_times = [0.0, 0.25, 0.5, 0.75, T_end]
    nearest_j = [int(np.argmin(np.abs(t_grid - rt))) for rt in rep_times]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, (rt, j) in enumerate(zip(rep_times, nearest_j)):
        if i >= len(axes):
            break
        vals = per_class_mean[:, j]
        pos = vals[vals > 0]
        ax = axes[i]
        if pos.size >= 2 and pos.max() / (pos.min() + 1e-30) > 10:
            bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()),
                               min(20, max(4, C // 2)))
            ax.hist(pos, bins=bins, color="steelblue",
                    edgecolor="black", alpha=0.8)
            ax.set_xscale("log")
        else:
            ax.hist(vals, bins=min(20, max(4, C // 2)),
                    color="steelblue", edgecolor="black", alpha=0.8)
        ax.axvline(vals.mean(), color="red", linestyle="--",
                   label=f"mean={vals.mean():.3g}")
        ax.axvline(np.median(vals), color="green", linestyle="--",
                   label=f"median={np.median(vals):.3g}")
        ax.axvline(np.quantile(vals, 0.10), color="grey", linestyle=":",
                   label=f"p10={np.quantile(vals, 0.10):.3g}")
        ax.axvline(np.quantile(vals, 0.90), color="grey", linestyle=":",
                   label=f"p90={np.quantile(vals, 0.90):.3g}")
        ax.set_title(f"t≈{rt}  (actual={t_grid[j]:.4f})")
        ax.set_xlabel("w_avg,c")
        ax.set_ylabel("classes")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    # ECDF panel in last slot
    ax = axes[-1]
    for i, (rt, j) in enumerate(zip(rep_times, nearest_j)):
        v = np.sort(per_class_mean[:, j])
        y = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, y, marker=".", label=f"t≈{rt}")
    ax.set_xscale("log")
    ax.set_xlabel("w_avg,c (log)")
    ax.set_ylabel("ECDF")
    ax.set_title("Class ECDF")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "wavg_class_distribution_selected_times.png"),
                dpi=140)
    plt.close()

    # -------- Per-time statistics CSV -------------------------------------
    with open(os.path.join(out_dir, "wavg_class_statistics_by_time.csv"), "w") as f:
        f.write("t,mean,median,std,iqr,min,max,cv,q10,q90,max_class_share\n")
        for j in range(T_rep):
            v = per_class_mean[:, j]
            iqr = float(np.subtract(*np.quantile(v, [0.75, 0.25])))
            max_share = float(v.max() / (v.sum() + 1e-30))
            f.write(f"{t_grid[j]:.10f},{v.mean():.10e},{np.median(v):.10e},"
                    f"{v.std():.10e},{iqr:.10e},{v.min():.10e},{v.max():.10e},"
                    f"{(v.std() / (v.mean() + 1e-30)):.10e},"
                    f"{np.quantile(v, 0.10):.10e},"
                    f"{np.quantile(v, 0.90):.10e},{max_share:.10e}\n")

    # -------- Per-class summary CSV ---------------------------------------
    with open(os.path.join(out_dir, "wavg_class_summary.csv"), "w") as f:
        f.write("class_id,class_name,integral_A_c,max,argmax_t,"
                "mean_early,mean_mid,mean_late,early_late_ratio,M_c\n")
        early_hi = 0.25 * T_end
        late_lo  = 0.75 * T_end
        for c in range(C):
            v = per_class_mean[c]
            A_int = float(_np_trapezoid(v, t_grid))
            j_max = int(np.argmax(v))
            early = t_grid <= early_hi
            mid   = (t_grid > early_hi) & (t_grid <= late_lo)
            late  = t_grid > late_lo
            m_e = float(v[early].mean()) if early.any() else float("nan")
            m_m = float(v[mid].mean())   if mid.any()   else float("nan")
            m_l = float(v[late].mean())  if late.any()  else float("nan")
            ratio = m_e / (m_l + 1e-30) if late.any() else float("nan")
            f.write(f"{c},{selected_class_ids[c]},{A_int:.10e},"
                    f"{v.max():.10e},{t_grid[j_max]:.10f},"
                    f"{m_e:.10e},{m_m:.10e},{m_l:.10e},{ratio:.10e},"
                    f"{per_class_meta[c]['M_c']}\n")

    # -------- Ranking plot -------------------------------------------------
    A_int_all = np.array([float(_np_trapezoid(per_class_mean[c], t_grid))
                          for c in range(C)])
    order_r = np.argsort(A_int_all)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.25 * C)))
    ax.barh(range(C), A_int_all[order_r], color="steelblue")
    ax.set_yticks(range(C))
    ax.set_yticklabels([selected_class_ids[i] for i in order_r], fontsize=7)
    ax.set_xlabel("integrated w_avg(t)")
    ax.set_xscale("log")
    ax.set_title("Class ranking by integrated amplification (log)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "wavg_class_integral_ranking.png"), dpi=140)
    plt.close()

    # -------- Uncertainty decomposition -----------------------------------
    # Var_probe   : mean over classes of  mean_traj Var_k w_traj_probe
    # Var_traj    : mean over classes of  Var_traj mean_k w_traj_probe
    # Var_class   : Var_class mean_traj mean_k
    var_probe = per_class_probe_var_mean.mean(axis=0)               # (T,)
    var_traj  = per_class_traj_mean.var(axis=1, ddof=1).mean(axis=0)
    var_class = per_class_mean.var(axis=0, ddof=1)

    sd_probe = np.sqrt(np.clip(var_probe, 0, None))
    sd_traj  = np.sqrt(np.clip(var_traj,  0, None))
    sd_class = np.sqrt(np.clip(var_class, 0, None))
    se_class = sd_class / max(1.0, math.sqrt(C))

    with open(os.path.join(out_dir, "wavg_uncertainty_decomposition.csv"), "w") as f:
        f.write("t,var_probe,var_trajectory,var_class,"
                "sd_probe,sd_trajectory,sd_class,se_class\n")
        for j in range(T_rep):
            f.write(f"{t_grid[j]:.10f},"
                    f"{var_probe[j]:.10e},{var_traj[j]:.10e},{var_class[j]:.10e},"
                    f"{sd_probe[j]:.10e},{sd_traj[j]:.10e},{sd_class[j]:.10e},"
                    f"{se_class[j]:.10e}\n")

    # Two panels: variances (top) and SDs/SE (bottom).
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax1.plot(t_grid, var_probe, marker="o", label="Var_probe")
    ax1.plot(t_grid, var_traj,  marker="s", label="Var_trajectory")
    ax1.plot(t_grid, var_class, marker="^", label="Var_class")
    ax1.set_yscale("log")
    ax1.set_ylabel("variance (log)")
    ax1.set_title("Uncertainty decomposition (variances)")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend()

    ax2.plot(t_grid, sd_probe, marker="o", label="SD_probe")
    ax2.plot(t_grid, sd_traj,  marker="s", label="SD_trajectory")
    ax2.plot(t_grid, sd_class, marker="^", label="SD_class")
    ax2.plot(t_grid, se_class, marker="v", label="SE_class")
    ax2.set_yscale("log")
    ax2.set_xlabel(f"t   (T_end = {T_end:g})")
    ax2.set_ylabel("std / SE (log)")
    ax2.set_title("Uncertainty decomposition (SDs / SE)")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "wavg_uncertainty_decomposition.png"), dpi=140)
    plt.close()

    # -------- Outlier report (any 20% contribution flagged) ---------------
    outlier_rows = []
    for j in range(T_rep):
        # Class-level: which class contributes > 20% of the sum at time t_j?
        contrib = per_class_mean[:, j] / (per_class_mean[:, j].sum() + 1e-30)
        for c in range(C):
            if contrib[c] > 0.20:
                outlier_rows.append({
                    "t": float(t_grid[j]),
                    "level": "class",
                    "class_id": c,
                    "class_name": selected_class_ids[c],
                    "share": float(contrib[c]),
                    "value": float(per_class_mean[c, j]),
                })
        # Trajectory-level: within each class, does any trajectory contribute > 20% of the class mean?
        for c in range(C):
            traj_vals = per_class_traj_mean[c, :, j]
            total = traj_vals.sum()
            if total <= 0:
                continue
            share = traj_vals / (total + 1e-30)
            argmax = int(np.argmax(share))
            if share[argmax] > 0.20:
                outlier_rows.append({
                    "t": float(t_grid[j]),
                    "level": "trajectory",
                    "class_id": c,
                    "class_name": selected_class_ids[c],
                    "trajectory_id": argmax,
                    "share": float(share[argmax]),
                    "value": float(traj_vals[argmax]),
                })
    with open(os.path.join(out_dir, "outlier_report.csv"), "w") as f:
        f.write("t,level,class_id,class_name,trajectory_id,share,value\n")
        for r in outlier_rows:
            f.write(f"{r['t']:.10f},{r['level']},{r['class_id']},{r['class_name']},"
                    f"{r.get('trajectory_id','')},{r['share']:.6e},{r['value']:.10e}\n")

    # -------- Trajectory-dominance per (c, t): top-1 and top-5 shares -----
    top1 = np.zeros((C, T_rep))
    top5 = np.zeros((C, T_rep))
    for c in range(C):
        for j in range(T_rep):
            v = per_class_traj_mean[c, :, j]
            total = v.sum()
            if total <= 0:
                continue
            sv = np.sort(v)[::-1]
            top1[c, j] = sv[0] / (total + 1e-30)
            top5[c, j] = sv[: min(5, len(sv))].sum() / (total + 1e-30)
    with open(os.path.join(out_dir, "trajectory_dominance.csv"), "w") as f:
        f.write("class_id,class_name," +
                ",".join(f"top1_t={t_grid[j]:.4f}" for j in range(T_rep)) + "," +
                ",".join(f"top5_t={t_grid[j]:.4f}" for j in range(T_rep)) + "\n")
        for c in range(C):
            row1 = ",".join(f"{top1[c, j]:.4e}" for j in range(T_rep))
            row5 = ",".join(f"{top5[c, j]:.4e}" for j in range(T_rep))
            f.write(f"{c},{selected_class_ids[c]},{row1},{row5}\n")

    # -------- Master npz ---------------------------------------------------
    np.savez(
        os.path.join(out_dir, "wavg_raw.npz"),
        times=t_grid,
        T_end=np.float64(T_end),
        selected_class_ids=np.array(selected_class_ids),
        class_mean=per_class_mean,
        class_trajectory_mean=per_class_traj_mean,
        class_probe_var_mean=per_class_probe_var_mean,
        global_mean=global_mean, global_median=global_median,
        global_q10=q10, global_q90=q90,
        global_ci_lower=ci_lo, global_ci_upper=ci_hi,
        trajectory_top1_share=top1, trajectory_top5_share=top5,
    )

    with open(os.path.join(out_dir, "class_id_to_name.json"), "w") as f:
        json.dump({str(i): selected_class_ids[i] for i in range(C)}, f, indent=2)


# ============================================================
# Microbenchmark
# ============================================================

@torch.no_grad()
def microbenchmark(Y: torch.Tensor, N: int, K: int, S: int, T_end: float,
                   device: torch.device, num_iters: int = 5) -> dict:
    M, D = Y.shape
    y_sq = (Y * Y).sum(dim=1, keepdim=True).T
    Z = torch.randn((N, D), device=device, dtype=Y.dtype)
    t = 0.5 * T_end

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    for _ in range(2):
        _v, alpha, mu = velocity(Z, Y, y_sq, t)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    v_times = []
    for _ in range(num_iters):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.time()
        v, alpha, mu = velocity(Z, Y, y_sq, t)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        v_times.append((time.time() - t0) * 1000.0)

    A = torch.randn((K, N, D), device=device, dtype=Y.dtype) / math.sqrt(D)
    jtv_times = []
    for _ in range(num_iters):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.time()
        _ = apply_JT_batched(A, Y, t, alpha, mu)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        jtv_times.append((time.time() - t0) * 1000.0)

    return {
        "M_c": int(M), "N": int(N), "K": int(K), "S": int(S),
        "T_end": float(T_end),
        "v_ms":   {"median": float(np.median(v_times)),
                   "p90": float(np.quantile(v_times, 0.9)),
                   "p99": float(np.quantile(v_times, 0.99))},
        "jtv_ms": {"median": float(np.median(jtv_times)),
                   "p90": float(np.quantile(jtv_times, 0.9)),
                   "p99": float(np.quantile(jtv_times, 0.99))},
        "predicted_forward_sec_per_class":
            float(S * np.median(v_times) / 1000.0),
        "predicted_backward_sec_per_class":
            float(S * np.median(jtv_times) / 1000.0),
    }


# ============================================================
# Main
# ============================================================

def _parse_dtype(name: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp64": torch.float64,
        "float64": torch.float64,
    }[str(name).lower()]


def _build_report_indices(cfg_flow: dict, S: int) -> list:
    """Prefer explicit report_indices; else use report_fractions * S.

    All indices are integers in [0, S]; must include 0 and S (T_end).
    """
    if "report_indices" in cfg_flow and cfg_flow["report_indices"] is not None:
        idx = [int(x) for x in cfg_flow["report_indices"]]
    elif "report_fractions" in cfg_flow and cfg_flow["report_fractions"] is not None:
        fracs = [float(x) for x in cfg_flow["report_fractions"]]
        idx = [int(round(f * S)) for f in fracs]
    else:
        idx = list(np.linspace(0, S, 16, dtype=int).tolist())

    # Deduplicate, sort, force endpoints.
    idx = sorted(set(idx))
    if idx[0] != 0:
        idx = [0] + idx
    if idx[-1] != S:
        idx = idx + [S]
    for x in idx:
        if x < 0 or x > S:
            raise ValueError(f"report index out of range: {x} (S={S})")
    return idx


def main():
    parser = argparse.ArgumentParser(
        description="w_avg(T, t) on the analytic ImageNet marginal flow (v2)"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    cli = parser.parse_args()

    cfg = load_config(cli.config)

    default_device = cli.device or cfg.get("device", "cuda:0")
    rank, world_size, device_str = setup_dist(default_device)
    device = torch.device(device_str)

    out_root = cli.output or cfg.get("output_dir") \
        or os.path.join(cfg.get("experiments_base_dir", "./experiment"),
                        cfg["experiment"]["name"])
    if rank == 0:
        os.makedirs(out_root, exist_ok=True)
        os.makedirs(os.path.join(out_root, "per_class"), exist_ok=True)
        shutil.copy(cli.config, os.path.join(out_root, "resolved_config.yaml"))
    barrier(world_size)

    log_path = os.path.join(out_root, f"progress.rank{rank}.jsonl")
    logger = JsonlLogger(log_path, rank, world_size,
                         echo=bool(cfg.get("logging", {}).get("echo_stdout", True)))

    seed = int(cfg["experiment"].get("seed", 2026))
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed + rank * 101)

    # ---- data / latent-mode config ---------------------------------------
    data_cfg = cfg["data"]
    latent_root = data_cfg["latent_root"]
    num_test_classes = int(data_cfg["num_test_classes"])
    latent_mode  = str(data_cfg.get("latent_mode", "fixed_posterior_sample"))
    latent_scale = float(data_cfg.get("latent_scale", 1.0))
    latent_shape = tuple(data_cfg.get("latent_shape",
                                      [DEFAULT_LATENT_C, DEFAULT_LATENT_H, DEFAULT_LATENT_W]))
    latent_D = int(data_cfg.get("latent_dimension",
                                latent_shape[0] * latent_shape[1] * latent_shape[2]))

    # ---- estimator / flow config -----------------------------------------
    T_end = float(cfg["flow"].get("terminal_time", 0.99))
    S = int(cfg["flow"]["solver_steps"])
    report_indices = _build_report_indices(cfg["flow"], S)
    t_grid_actual = np.array(report_indices, dtype=np.float64) * (T_end / S)

    N = int(cfg["estimator"]["samples_per_class"])
    K = int(cfg["estimator"]["probes_per_sample"])
    precision = str(cfg["estimator"].get("precision", "fp32"))
    dtype = _parse_dtype(precision)
    bootstrap_replicates = int(cfg["estimator"].get("bootstrap_replicates", 10000))
    progress_interval = int(cfg["logging"].get("progress_interval_steps", 8))

    if T_end >= 1.0:
        raise ValueError(f"flow.terminal_time must be < 1 (got {T_end}). "
                         "Integrating exactly to t=1 is what the revision doc "
                         "explicitly forbids.")

    all_classes = list_packed_classes(latent_root)
    if rank == 0:
        print(f"Found {len(all_classes)} classes at {latent_root}")
        print(f"T_end={T_end}  S={S}  h=T/S={T_end/S:.6e}  "
              f"N={N}  K={K}  precision={precision}  "
              f"latent_mode={latent_mode}  latent_scale={latent_scale}")

    rng = np.random.default_rng(seed)
    sel_idx = rng.choice(len(all_classes), size=num_test_classes, replace=False)
    sel_idx = sorted(sel_idx.tolist())
    selected_classes = [all_classes[i] for i in sel_idx]

    if rank == 0:
        with open(os.path.join(out_root, "selected_classes.json"), "w") as f:
            json.dump({
                "seed": seed,
                "num_available_classes": len(all_classes),
                "num_test_classes": num_test_classes,
                "selected_indices": sel_idx,
                "selected_class_names": selected_classes,
                "T_end": T_end, "S": S,
                "actual_times": t_grid_actual.tolist(),
                "latent_mode": latent_mode,
                "latent_scale": latent_scale,
                "precision": precision,
            }, f, indent=2)

    local_class_indices = [c for c in range(num_test_classes) if c % world_size == rank]
    if rank == 0:
        print(f"Class sharding: each rank handles ~{len(local_class_indices)} classes.")
    logger.log({
        "phase": "setup",
        "world_size": world_size,
        "T_end": T_end, "S": S,
        "num_test_classes": num_test_classes,
        "local_class_count": len(local_class_indices),
        "seed": seed, "precision": precision,
        "latent_mode": latent_mode, "latent_scale": latent_scale,
    })

    # ---- microbenchmark ---------------------------------------------------
    if rank == 0:
        print("\n[microbenchmark] loading class 0 on rank 0...")
        Y_probe = load_class_latents(
            latent_root, selected_classes[0], device,
            seed=seed + sel_idx[0],
            mode=latent_mode, scale=latent_scale, dtype=dtype,
        )
        bench = microbenchmark(Y_probe, N=N, K=K, S=S, T_end=T_end, device=device)
        pred_total_per_class = (bench["predicted_forward_sec_per_class"]
                                + bench["predicted_backward_sec_per_class"])
        classes_per_rank = math.ceil(num_test_classes / world_size)
        bench["predicted_total_run_seconds"] = float(pred_total_per_class * classes_per_rank)
        bench["classes_per_rank_worst_case"] = int(classes_per_rank)
        with open(os.path.join(out_root, "runtime_profile.json"), "w") as f:
            json.dump(bench, f, indent=2)
        print(f"[microbenchmark] v_ms(p50)={bench['v_ms']['median']:.2f}   "
              f"jtv_ms(p50)={bench['jtv_ms']['median']:.2f}")
        print(f"[microbenchmark] predicted total run ≈ "
              f"{fmt_time(bench['predicted_total_run_seconds'])} (worst rank).")
        del Y_probe
        if device.type == "cuda":
            torch.cuda.empty_cache()
    barrier(world_size)

    # ---- iterate over local classes ---------------------------------------
    resume_completed = set()
    for c_global in local_class_indices:
        ckpt_path = os.path.join(out_root, "per_class",
                                 f"class_{c_global:04d}.npz")
        if os.path.exists(ckpt_path):
            resume_completed.add(c_global)

    if resume_completed and rank == 0:
        print(f"[resume] found {len(resume_completed)} already-done classes.")

    run_t0 = time.time()

    for local_i, c_global in enumerate(local_class_indices):
        ckpt_path = os.path.join(out_root, "per_class",
                                 f"class_{c_global:04d}.npz")
        if c_global in resume_completed:
            continue

        cls_name = selected_classes[c_global]
        logger.log({
            "phase": "load_class",
            "class_global": c_global,
            "class_local": local_i + 1,
            "class_name": cls_name,
        }, stdout_msg=(f"[rank {rank}] loading class {c_global:03d} ({cls_name}) "
                       f"local {local_i + 1}/{len(local_class_indices)}"))

        class_seed = seed * 10007 + sel_idx[c_global]
        Y = load_class_latents(
            latent_root, cls_name, device,
            seed=class_seed, mode=latent_mode, scale=latent_scale, dtype=dtype,
        )

        cls_t0 = time.time()
        try:
            result = run_class(
                Y=Y, class_seed=class_seed + 1,
                N=N, K=K, S=S, T_end=T_end,
                report_indices=report_indices,
                device=device, logger=logger, rank=rank,
                class_local_idx=local_i + 1,
                class_local_total=len(local_class_indices),
                class_global_idx=c_global + 1,
                class_global_total=num_test_classes,
                progress_interval=progress_interval,
                dtype=dtype,
            )
        except RuntimeError as e:
            logger.log({"phase": "abort", "class_global": c_global,
                        "class_name": cls_name, "error": str(e)},
                       stdout_msg=f"[rank {rank}] ABORT class {c_global}: {e}")
            raise

        cls_time = time.time() - cls_t0

        tmp_path = ckpt_path + f".tmp.{os.getpid()}.npz"
        np.savez(
            tmp_path,
            class_mean=result["class_mean"],
            class_traj_mean=result["class_traj_mean"],
            class_traj_probe=result["class_traj_probe"],
            probe_var_by_traj=result["probe_var_by_traj"],
            probe_var_mean=result["probe_var_mean"],
            M_c=result["M_c"], N=result["N"], K=result["K"], S=result["S"],
            T_end=result["T_end"],
            report_indices=np.array(result["report_indices"]),
            actual_times=np.array(result["actual_times"]),
            class_name=cls_name, class_global=c_global,
            fwd_time_sec=result["fwd_time_sec"],
            adj_time_sec=result["adj_time_sec"],
            mean_v_batch_ms=result["mean_v_batch_ms"],
            mean_jtv_batch_ms=result["mean_jtv_batch_ms"],
            peak_gpu_mem_gib=result["peak_gpu_mem_gib"],
            resp_entropy_mean=result["resp_entropy_mean"],
            resp_neff_mean=result["resp_neff_mean"],
            resp_sumcheck_max_dev=result["resp_sumcheck_max_dev"],
            latent_mode=latent_mode,
            latent_scale=latent_scale,
            precision=precision,
        )
        os.replace(tmp_path, ckpt_path)

        elapsed = time.time() - run_t0
        eta_total = elapsed * (len(local_class_indices) - (local_i + 1)) \
            / max(1, local_i + 1)
        logger.log({
            "phase": "class_done",
            "class_global": c_global,
            "class_local": local_i + 1,
            "class_name": cls_name, "M_c": result["M_c"],
            "runtime_sec": round(cls_time, 3),
            "mean_v_batch_ms":   result["mean_v_batch_ms"],
            "mean_jtv_batch_ms": result["mean_jtv_batch_ms"],
            "peak_gpu_mem_gib":  result["peak_gpu_mem_gib"],
        }, stdout_msg=(
            f"[rank {rank}] class_done={c_global + 1:03d}/{num_test_classes:02d} "
            f"M_c={result['M_c']} runtime_sec={cls_time:.2f} "
            f"mean_v_batch_ms={result['mean_v_batch_ms']:.2f} "
            f"mean_jtv_batch_ms={result['mean_jtv_batch_ms']:.2f} "
            f"peak_gpu_mem_gib={result['peak_gpu_mem_gib']:.2f} "
            f"elapsed={fmt_time(elapsed)} eta_rank={fmt_time(eta_total)}"
        ))

        del Y
        if device.type == "cuda":
            torch.cuda.empty_cache()

    barrier(world_size)

    # ============================================================
    # Aggregation (rank 0)
    # ============================================================
    if rank == 0:
        print("\n[aggregate] loading per-class checkpoints...")
        T_rep = len(report_indices)
        C = num_test_classes
        per_class_mean = np.zeros((C, T_rep), dtype=np.float64)
        per_class_traj_mean = np.zeros((C, N, T_rep), dtype=np.float64)
        per_class_probe_var_mean = np.zeros((C, T_rep), dtype=np.float64)
        per_class_meta = [{} for _ in range(C)]
        missing = []
        for c_global in range(C):
            ckpt_path = os.path.join(out_root, "per_class",
                                     f"class_{c_global:04d}.npz")
            if not os.path.exists(ckpt_path):
                missing.append(c_global)
                continue
            data = np.load(ckpt_path)
            per_class_mean[c_global] = data["class_mean"]
            per_class_traj_mean[c_global] = data["class_traj_mean"]
            per_class_probe_var_mean[c_global] = data["probe_var_mean"]
            per_class_meta[c_global] = {
                "M_c": int(data["M_c"]), "class_name": str(data["class_name"]),
                "fwd_time_sec": float(data["fwd_time_sec"]),
                "adj_time_sec": float(data["adj_time_sec"]),
                "mean_v_batch_ms":   float(data["mean_v_batch_ms"]),
                "mean_jtv_batch_ms": float(data["mean_jtv_batch_ms"]),
                "peak_gpu_mem_gib":  float(data["peak_gpu_mem_gib"]),
                "resp_entropy_mean": float(data["resp_entropy_mean"]),
                "resp_neff_mean":    float(data["resp_neff_mean"]),
                "resp_sumcheck_max_dev": float(data["resp_sumcheck_max_dev"]),
            }
        if missing:
            raise RuntimeError(
                f"Missing per-class checkpoints for indices: {missing}. "
                f"Re-run to complete these classes (resume is supported)."
            )

        write_outputs(
            out_dir=out_root,
            t_grid=t_grid_actual,
            T_end=T_end,
            selected_class_ids=selected_classes,
            per_class_mean=per_class_mean,
            per_class_traj_mean=per_class_traj_mean,
            per_class_probe_var_mean=per_class_probe_var_mean,
            per_class_meta=per_class_meta,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )

        # Numerical health
        max_dev = max(m["resp_sumcheck_max_dev"] for m in per_class_meta if m)
        terminal_pos = report_indices.index(S)
        terminal_w = float(per_class_mean[:, terminal_pos].mean())
        health = {
            "T_end": T_end, "S": S,
            "terminal_w_avg_at_T_end": terminal_w,
            "terminal_deviation_from_1": abs(terminal_w - 1.0),
            "max_softmax_sum_deviation": max_dev,
            "num_classes": C,
            "any_nan_or_inf": False,
            "latent_mode": latent_mode,
            "latent_scale": latent_scale,
            "precision": precision,
            "seed": seed,
            "actual_times": t_grid_actual.tolist(),
        }
        with open(os.path.join(out_root, "numerical_health.json"), "w") as f:
            json.dump(health, f, indent=2)

        print(f"\n[done] Terminal w_avg(T, T) = {terminal_w:.6f} (should be ≈ 1)")
        print(f"[done] Outputs in {out_root}")

    barrier(world_size)

    # ============================================================
    # Validation subruns (rank 0)
    # ============================================================
    val_cfg = cfg.get("validation", {})
    do_val = bool(val_cfg.get("enabled", True))

    if do_val and rank == 0:
        print("\n[validation] running solver convergence, autograd VJP, "
              "FD trace, marginal moments on rank 0...")
        val_classes  = int(val_cfg.get("num_classes", 4))
        val_class_globals = list(range(min(val_classes, num_test_classes)))
        solver_steps_list = list(val_cfg.get("solver_steps", [128, 256, 512]))
        fd_delta_list = list(val_cfg.get("finite_difference_delta",
                                         val_cfg.get("finite_difference_eta",
                                                     [1.0e-4, 1.0e-3])))
        N_val = int(val_cfg.get("val_N", 8))
        K_val = int(val_cfg.get("val_K", 16))
        vjp_dtype_name = str(val_cfg.get("autograd_vjp_dtype", precision))
        vjp_dtype = _parse_dtype(vjp_dtype_name)
        do_marginal = bool(val_cfg.get("marginal_moments", True))
        do_autograd = bool(val_cfg.get("autograd_vjp", True))

        conv_records = []
        jvp_records = []
        moment_records = []
        vjp_records = []

        # Common target grid for continuous-time alignment.
        target_times = np.linspace(0.0, T_end, 16, dtype=np.float64)

        for c_val in val_class_globals:
            cls_name = selected_classes[c_val]
            class_seed = seed * 10007 + sel_idx[c_val]
            Y = load_class_latents(
                latent_root, cls_name, device,
                seed=class_seed, mode=latent_mode, scale=latent_scale,
                dtype=dtype,
            )
            if vjp_dtype != dtype:
                Y_vjp = Y.to(vjp_dtype)
            else:
                Y_vjp = Y

            # ---- solver convergence at fixed T (aligned in continuous time) ---
            wavg_curves = {}
            for S_alt in solver_steps_list:
                rep_alt = list(np.linspace(0, S_alt, 16, dtype=int).tolist())
                rep_alt[0] = 0
                rep_alt[-1] = S_alt
                res = run_class(
                    Y=Y, class_seed=class_seed + 2,
                    N=N_val, K=K_val, S=S_alt, T_end=T_end,
                    report_indices=rep_alt,
                    device=device, logger=logger, rank=0,
                    class_local_idx=1, class_local_total=1,
                    class_global_idx=c_val + 1,
                    class_global_total=len(val_class_globals),
                    progress_interval=0,
                    resp_stats_every=0, latency_probe_interval=0,
                    dtype=dtype,
                )
                actual_t = np.array(res["actual_times"], dtype=np.float64)
                aligned = np.interp(target_times, actual_t, res["class_mean"])
                wavg_curves[S_alt] = aligned

            S_ref = max(solver_steps_list)
            for S_alt in solver_steps_list:
                if S_alt == S_ref:
                    continue
                rel_err = np.abs(wavg_curves[S_alt] - wavg_curves[S_ref]) \
                    / (np.abs(wavg_curves[S_ref]) + 1e-12)
                conv_records.append({
                    "class_id": c_val, "class_name": cls_name,
                    "S": S_alt, "S_ref": S_ref,
                    "max_rel_err":  float(rel_err.max()),
                    "mean_rel_err": float(rel_err.mean()),
                })

            # ---- trace-averaged FD at midpoint ------------------------------
            s_mid = S // 2
            r_fd = trace_averaged_fd_validation(
                Y=Y, Y_fp64=Y.to(torch.float64),
                class_seed=class_seed + 3,
                N_val=N_val, K_val=K_val,
                S=S, T_end=T_end, solver_index=s_mid,
                delta_list=fd_delta_list, device=device, dtype=dtype,
            )
            r_fd.update({"class_id": c_val, "class_name": cls_name})
            jvp_records.append(r_fd)

            # ---- autograd VJP local check ---------------------------------
            if do_autograd:
                for t_frac in (0.25, 0.5, 0.75):
                    r_vjp = autograd_vjp_validation(
                        Y_vjp, t=t_frac * T_end,
                        N_val=min(4, N_val), K_val=min(4, K_val),
                        device=device, seed=class_seed + 5,
                        dtype=vjp_dtype,
                    )
                    r_vjp.update({"class_id": c_val, "class_name": cls_name})
                    vjp_records.append(r_vjp)

            # ---- marginal-moment validation --------------------------------
            if do_marginal:
                r_mom = marginal_moment_validation(
                    Y=Y, class_seed=class_seed + 7,
                    N_val=max(64, N_val * 2), S=S, T_end=T_end,
                    report_indices=report_indices, device=device, dtype=dtype,
                )
                r_mom.update({"class_id": c_val, "class_name": cls_name})
                moment_records.append(r_mom)

            del Y, Y_vjp
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # ---- write validation CSVs / plots --------------------------------
        with open(os.path.join(out_root, "solver_convergence.csv"), "w") as f:
            f.write("class_id,class_name,S,S_ref,max_rel_err,mean_rel_err\n")
            for r in conv_records:
                f.write(f"{r['class_id']},{r['class_name']},{r['S']},"
                        f"{r['S_ref']},{r['max_rel_err']:.6e},"
                        f"{r['mean_rel_err']:.6e}\n")

        with open(os.path.join(out_root, "fd_trace_validation.csv"), "w") as f:
            hdr = ",".join(f"rel_err_delta_{d:.0e}" for d in fd_delta_list)
            f.write(f"class_id,class_name,solver_index,t,w_back,{hdr}\n")
            for r in jvp_records:
                errs = ",".join(f"{r['fd_by_delta'][float(d)]['rel_err']:.6e}"
                                for d in fd_delta_list)
                f.write(f"{r['class_id']},{r['class_name']},"
                        f"{r['solver_index']},{r['t']:.6f},"
                        f"{r['w_back']:.6e},{errs}\n")

        if do_autograd:
            with open(os.path.join(out_root, "autograd_vjp_validation.csv"), "w") as f:
                f.write("class_id,class_name,t,dtype,"
                        "rel_error_analytic_vs_autograd,"
                        "analytic_norm,autograd_norm\n")
                for r in vjp_records:
                    f.write(f"{r['class_id']},{r['class_name']},{r['t']:.6f},"
                            f"{r['dtype']},"
                            f"{r['rel_error_analytic_vs_autograd']:.6e},"
                            f"{r['analytic_norm']:.6e},"
                            f"{r['autograd_norm']:.6e}\n")

        if do_marginal:
            with open(os.path.join(out_root, "marginal_moment_validation.csv"), "w") as f:
                hdr = ",".join(f"t={t:.6f}"
                               for t in moment_records[0]["actual_times"])
                f.write("class_id,class_name,metric," + hdr + "\n")
                for r in moment_records:
                    row_mean = ",".join(f"{v:.6e}" for v in r["mean_rel_err"])
                    row_cov  = ",".join(f"{v:.6e}" for v in r["cov_trace_rel_err"])
                    f.write(f"{r['class_id']},{r['class_name']},mean_rel_err,{row_mean}\n")
                    f.write(f"{r['class_id']},{r['class_name']},cov_trace_rel_err,{row_cov}\n")

        # ---- simple convergence plot -------------------------------------
        plt = _lazy_plt()
        if conv_records:
            fig, ax = plt.subplots(figsize=(8, 5))
            for S_alt in solver_steps_list:
                errs = [r["max_rel_err"] for r in conv_records if r["S"] == S_alt]
                if errs:
                    ax.plot(range(len(errs)), errs, marker="o", label=f"S={S_alt}")
            ax.set_yscale("log")
            ax.set_xlabel("validation class")
            ax.set_ylabel(f"max relative error vs S={max(solver_steps_list)}")
            ax.set_title("Solver-convergence relative error at fixed T")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_root, "wavg_solver_relative_error.png"),
                        dpi=140)
            plt.close()

        print("[validation] wrote solver_convergence.csv, fd_trace_validation.csv, "
              "autograd_vjp_validation.csv, marginal_moment_validation.csv")

    barrier(world_size)
    logger.close()
    if world_size > 1:
        dist.destroy_process_group()

    if rank == 0:
        print(f"\nAll done. Results at: {out_root}")


if __name__ == "__main__":
    main()
