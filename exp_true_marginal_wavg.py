"""
exp_true_marginal_wavg.py -- Estimate w_avg(t) on the analytic class-conditional
                             marginal flow of the empirical ImageNet latent
                             distribution.

Given a class c with target latents {y_i^(c)}_{i=1..M_c}, prior z_0 ~ N(0,I),
and the linear interpolant  z_t = (1-t) z_0 + t y_i,  the induced marginal
velocity is analytic:

    alpha_i(z, t | c) = softmax_i( -||z - t y_i||^2 / (2 (1-t)^2) )
    mu_c(z, t)        = sum_i alpha_i y_i
    v_c(z, t)         = ( mu_c(z, t) - z ) / (1 - t)

We integrate z with S=256 Euler steps to produce a trajectory z_0, ..., z_S
(no velocity query at t=1).  For each trajectory we then run a *backward
adjoint sweep* using K unit-norm Rademacher probes anchored at the terminal
time; this yields w_avg estimates at all 16 reporting times in one pass.

The Euler-step Jacobian is

    G_n = I + h J_c(z_n, t_n),
    J_c = -1/(1-t) I + t / (1-t)^3 * C_alpha,
    C_alpha = sum_i alpha_i (y_i - mu)(y_i - mu)^T.

We never form C_alpha or J explicitly. Instead we apply J^T a via two
responsibility-weighted matvecs of size (M_c, d):

    J^T a = c1 * a  +  c2 * ( Y^T ( alpha * (Y a - <mu, a> 1) )  -  mu <mu, a> )
                                                                  <-- constant
    with c1 = -1/(1-t),  c2 = t / (1-t)^3.

This runs on 8 GPUs by sharding classes round-robin. Only the final
statistics require reduction; no cross-rank comm during compute.

Usage (multi-GPU):
    torchrun --standalone --nproc_per_node=8 exp_true_marginal_wavg.py \
        --config configs/true_marginal_wavg_imagenet.yaml \
        --output outputs/true_marginal_wavg_imagenet

Usage (single-GPU debug):
    python exp_true_marginal_wavg.py --config configs/... --device cuda:0
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import yaml


# ============================================================
# Constants (matches encode_dataset.py / train.py: 256x256 -> 4x32x32 latents)
# ============================================================

LATENT_C = 4
LATENT_H = 32
LATENT_W = 32
LATENT_D = LATENT_C * LATENT_H * LATENT_W          # 4096
VAE_SCALE = 0.18215


# ============================================================
# Distributed setup
# ============================================================

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
    if world_size > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


# ============================================================
# Config
# ============================================================

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ============================================================
# JSONL / progress logger
# ============================================================

class JsonlLogger:
    """Per-rank JSONL writer; also mirrors messages to stdout on rank 0."""

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
    """Return a sorted list of class names (WordNet IDs) available on disk."""
    if not os.path.isdir(latent_root):
        raise FileNotFoundError(f"latent_root does not exist: {latent_root}")
    files = sorted(f for f in os.listdir(latent_root) if f.endswith(".npy"))
    if not files:
        raise RuntimeError(f"No .npy files found under {latent_root}")
    return [os.path.splitext(f)[0] for f in files]


def load_class_latents_fixed(
    latent_root: str,
    class_name: str,
    device: torch.device,
    seed: int,
    flip: str = "original",
) -> torch.Tensor:
    """Load all pre-encoded latents of one ImageNet class into a (M_c, D) tensor.

    Per user decision (Q1): a single posterior sample epsilon is drawn per image
    and frozen. This gives a deterministic finite point cloud {y_i^(c)} matching
    the "empirical target distribution without endpoint smoothing" requirement.

    The .npy file has shape (M_c, 2, 2, 4, H, W) fp32:
        axis 1 (orientation) : 0 = original, 1 = horizontally flipped
        axis 2 (parameter)   : 0 = mean,     1 = std

    Args:
        latent_root : directory holding <class>.npy files.
        class_name  : e.g. "n01440764".
        device      : CUDA device to put the result on.
        seed        : deterministic epsilon seed (unique per class).
        flip        : "original" (default) or "flipped"; the default run
                      uses only the un-flipped orientation.

    Returns:
        Y : (M_c, LATENT_D) float32 tensor on `device`, already multiplied
            by the standard 0.18215 VAE scale (matches training input).
    """
    ori_idx = 0 if flip == "original" else 1
    path = os.path.join(latent_root, class_name + ".npy")
    arr = np.load(path, mmap_mode="r")                     # (M, 2, 2, 4, H, W)
    if arr.ndim != 6 or arr.shape[1] < ori_idx + 1 or arr.shape[2] != 2:
        raise RuntimeError(f"Unexpected packed-latent shape at {path}: {arr.shape}")
    mean_np = np.array(arr[:, ori_idx, 0], dtype=np.float32, copy=True)   # (M, 4, H, W)
    std_np  = np.array(arr[:, ori_idx, 1], dtype=np.float32, copy=True)
    del arr
    mean_t = torch.from_numpy(mean_np).to(device)
    std_t  = torch.from_numpy(std_np).to(device)

    gen = torch.Generator(device=device).manual_seed(int(seed))
    eps = torch.randn(mean_t.shape, device=device, dtype=torch.float32, generator=gen)
    y = (mean_t + std_t * eps) * VAE_SCALE                                 # (M, 4, H, W)
    return y.reshape(y.shape[0], -1).contiguous()                          # (M, D)


# ============================================================
# Analytic velocity + Jacobian-transpose action
# ============================================================

@torch.no_grad()
def responsibilities(
    Z: torch.Tensor,        # (N, D) states
    Y: torch.Tensor,        # (M, D) targets
    t: float,
) -> tuple:
    """Compute responsibilities alpha_i(z, t | c) for each state.

    Uses ||z - t y||^2 / (2 (1-t)^2). Log-softmax is shift-invariant so it
    stays numerically stable even for tiny (1-t).

    Returns:
        alpha : (N, M) float32; each row sums to 1.
        mu    : (N, D) float32; weighted target mean sum_i alpha_i y_i.
    """
    one_minus_t = 1.0 - t
    denom = 2.0 * (one_minus_t ** 2)

    # ||z - t y||^2 = ||z||^2 - 2 t <z, y> + t^2 ||y||^2
    z_sq = (Z * Z).sum(dim=1, keepdim=True)              # (N, 1)
    y_sq = (Y * Y).sum(dim=1, keepdim=True).T            # (1, M)
    zy   = Z @ Y.T                                       # (N, M)
    dist_sq = z_sq - 2.0 * t * zy + (t * t) * y_sq       # (N, M)
    logits = -dist_sq / denom                            # (N, M)

    alpha = torch.softmax(logits, dim=1)                 # stable shifted softmax
    mu = alpha @ Y                                       # (N, D)
    return alpha, mu


@torch.no_grad()
def velocity(Z: torch.Tensor, Y: torch.Tensor, t: float) -> tuple:
    """v_c(z, t) = (mu - z) / (1 - t). Returns (v, alpha, mu) for reuse."""
    alpha, mu = responsibilities(Z, Y, t)
    v = (mu - Z) / (1.0 - t)
    return v, alpha, mu


@torch.no_grad()
def apply_JT_batched(
    A: torch.Tensor,        # (K, N, D) probe adjoints per state (K probes, N traj)
    Z: torch.Tensor,        # (N, D)
    Y: torch.Tensor,        # (M, D)
    t: float,
    alpha: torch.Tensor,    # (N, M)
    mu: torch.Tensor,       # (N, D)
) -> torch.Tensor:
    """Apply J_c(z, t)^T to a batch of adjoints, without forming any (D, D) or
    even (M, D) intermediate per probe.

    J = -1/(1-t) I + t / (1-t)^3  * C_alpha
      with  C_alpha = sum_i alpha_i (y_i - mu)(y_i - mu)^T.

    Since C_alpha is symmetric, J is symmetric, and

        (C_alpha a) = sum_i alpha_i (y_i - mu) <y_i - mu, a>
                    = Y^T ( alpha * (Y a - <mu, a>) )  -  mu * ( <mu, a>_alpha )
    but the neat identity we actually use is
        C_alpha a = Y^T ( alpha * s )  -  mu * sum_i alpha_i s_i
                     with s_i = <y_i, a> - <mu, a>.
    Since sum_i alpha_i s_i = <mu, a> - <mu, a> = 0, the last term vanishes:
        C_alpha a = Y^T ( alpha * (Y a - <mu, a>) ).

    Args:
        A : (K, N, D). One D-dim adjoint per (probe, trajectory).
        Z : (N, D). Unused directly (kept for symmetry / debugging).
        Y : (M, D). Class targets.
        alpha, mu : outputs of `responsibilities` at (Z, t).

    Returns:
        JT_A : (K, N, D) float32.
    """
    _ = Z  # silence unused; kept to make call sites uniform
    one_minus_t = 1.0 - t
    c1 = -1.0 / one_minus_t                              # scalar
    c2 = t / (one_minus_t ** 3)                          # scalar

    K, N, D = A.shape
    M = Y.shape[0]

    # Reshape A to (K*N, D) for batched einsum with Y (M, D).
    A_flat = A.reshape(K * N, D)                         # (K*N, D)

    # <mu, a>_n,k  ->  (K*N,). We need per-(k, n) mu[n]. Broadcast mu -> (K, N, D).
    mu_dot_a = (A * mu.unsqueeze(0)).sum(dim=-1)         # (K, N)
    mu_dot_a_flat = mu_dot_a.reshape(K * N, 1)           # (K*N, 1)

    # (Y @ a)_i for each (k, n) -> (K*N, M).
    Ya = A_flat @ Y.T                                    # (K*N, M)

    # s = Y a  -  <mu, a>  broadcast over i.
    s = Ya - mu_dot_a_flat                               # (K*N, M)

    # weighted sum over i: (alpha_n * s_{kn}) then map back through Y.
    # alpha has shape (N, M); expand to (K, N, M) via broadcasting after reshape.
    alpha_flat = alpha.unsqueeze(0).expand(K, N, M).reshape(K * N, M)   # (K*N, M)
    weighted = alpha_flat * s                                            # (K*N, M)

    Ca = weighted @ Y                                    # (K*N, D)  = C_alpha a
    Ca = Ca.reshape(K, N, D)

    return c1 * A + c2 * Ca


# ============================================================
# Forward flow (analytic velocity) + adjoint sweep
# ============================================================

@torch.no_grad()
def run_class(
    Y: torch.Tensor,             # (M_c, D) targets
    class_seed: int,
    N: int,                       # trajectories per class
    K: int,                       # probes per trajectory
    S: int,                       # solver steps
    report_indices: list,         # list of int in [0, S]; must include 0 and S
    device: torch.device,
    logger: JsonlLogger,
    rank: int,
    class_local_idx: int,
    class_local_total: int,
    class_global_idx: int,
    class_global_total: int,
    progress_interval: int = 8,
    resp_stats_every: int = 32,
    latency_probe_interval: int = 32,
) -> dict:
    """Run one class: forward Euler flow + backward adjoint sweep.

    Returns a dict with per-class results and diagnostics.
    """
    M, D = Y.shape
    h = 1.0 / S

    # ---- allocate state cache (only at reporting indices to save memory) ----
    # Adjoint needs z_n at every step n=0..S-1, but we can regenerate them from
    # the deterministic forward pass to avoid storing all S+1 states. However
    # since D=4096 and N=64 => 64 * S * D * 4 bytes = 64*256*4096*4 = 256 MiB,
    # which is well within budget. Store all states.
    Z_cache = torch.empty((S + 1, N, D), dtype=torch.float32, device=device)

    # ---- initial states (seeded per class for reproducibility) ------------
    gen = torch.Generator(device=device).manual_seed(int(class_seed))
    Z_cache[0] = torch.randn((N, D), device=device, dtype=torch.float32, generator=gen)

    # ---- forward Euler flow -----------------------------------------------
    v_latency = []
    resp_entropy_samples = []
    resp_neff_samples = []
    resp_sumcheck_max_dev = 0.0
    fwd_t0 = time.time()

    for n in range(S):
        t = n / S
        if latency_probe_interval and (n % latency_probe_interval == 0):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_call0 = time.time()
        v, alpha, mu = velocity(Z_cache[n], Y, t)
        if latency_probe_interval and (n % latency_probe_interval == 0):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            v_latency.append((time.time() - t_call0) * 1000.0)

        # Optional: track responsibility stats at a subset of steps for QC.
        if resp_stats_every and (n % resp_stats_every == 0):
            a_sums = alpha.sum(dim=1)                          # (N,)
            resp_sumcheck_max_dev = max(
                resp_sumcheck_max_dev,
                float((a_sums - 1.0).abs().max().item()),
            )
            H_a = -(alpha * (alpha.clamp_min(1e-30)).log()).sum(dim=1)   # (N,)
            neff = 1.0 / (alpha * alpha).sum(dim=1).clamp_min(1e-30)     # (N,)
            resp_entropy_samples.append(float(H_a.mean().item()))
            resp_neff_samples.append(float(neff.mean().item()))

        Z_cache[n + 1] = Z_cache[n] + h * v

        if not torch.isfinite(Z_cache[n + 1]).all():
            raise RuntimeError(
                f"[rank {rank}] non-finite state at step {n} "
                f"(class local {class_local_idx}/{class_local_total})"
            )

        if rank == 0 and progress_interval and ((n + 1) % progress_interval == 0 or n == S - 1):
            elapsed = time.time() - fwd_t0
            eta = elapsed * (S - n - 1) / max(1, n + 1)
            v_ms = np.median(v_latency) if v_latency else float("nan")
            msg = (f"[rank {rank}] phase=forward "
                   f"class_global={class_global_idx:03d}/{class_global_total:02d} "
                   f"class_local={class_local_idx:03d}/{class_local_total:02d} "
                   f"M_c={M} sample=N={N} step={n + 1:03d}/{S} t={t:.6f} "
                   f"v_batch_ms={v_ms:.2f} elapsed={fmt_time(elapsed)} "
                   f"eta=NA eta_class={fmt_time(eta)}")
            logger.log({
                "phase": "forward", "class_global": class_global_idx,
                "class_local": class_local_idx, "step": n + 1, "S": S,
                "t": t, "M_c": M, "v_batch_ms": v_ms,
            }, stdout_msg=msg)

    fwd_time = time.time() - fwd_t0

    # ---- backward adjoint sweep --------------------------------------------
    # Draw K unit-norm Rademacher endpoint probes.
    q_gen = torch.Generator(device=device).manual_seed(int(class_seed) + 999983)
    r = torch.empty((K, N, D), device=device, dtype=torch.float32)
    r.bernoulli_(0.5, generator=q_gen).mul_(2.0).sub_(1.0)
    r.div_(math.sqrt(D))                                     # (K, N, D), each ||q||=1
    A = r                                                    # a_{S, k}

    # Cache ||a_n||^2 at every solver index. We fill entries as we sweep.
    # Shape: (len(report_indices), N, K).
    idx_to_pos = {s: i for i, s in enumerate(report_indices)}
    w_per_traj = torch.zeros((len(report_indices), N, K), dtype=torch.float64, device=device)

    # Terminal identity: at t=1 (n=S), ||a||^2 = ||q||^2 = 1 exactly.
    if S in idx_to_pos:
        w_per_traj[idx_to_pos[S]] = 1.0

    jtv_latency = []
    adj_t0 = time.time()

    # Backward: a_n = a_{n+1} + h * J(z_n, t_n)^T a_{n+1}, for n = S-1 down to 0.
    for n in range(S - 1, -1, -1):
        t = n / S
        # Recompute alpha, mu at (z_n, t). Small overhead vs re-storing.
        alpha, mu = responsibilities(Z_cache[n], Y, t)

        if latency_probe_interval and (n % latency_probe_interval == 0):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_call0 = time.time()
        JT_A = apply_JT_batched(A, Z_cache[n], Y, t, alpha, mu)   # (K, N, D)
        if latency_probe_interval and (n % latency_probe_interval == 0):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            jtv_latency.append((time.time() - t_call0) * 1000.0)

        A = A + h * JT_A                                     # a_n

        if not torch.isfinite(A).all():
            raise RuntimeError(
                f"[rank {rank}] non-finite adjoint at step {n} "
                f"(class local {class_local_idx}/{class_local_total})"
            )

        # Record if this solver index is a report time.
        if n in idx_to_pos:
            # ||a_n||^2 across dim D -> (K, N).  Store as (N, K).
            sq = (A.to(torch.float64) * A.to(torch.float64)).sum(dim=-1)   # (K, N)
            w_per_traj[idx_to_pos[n]] = sq.T                              # (N, K)

        if rank == 0 and progress_interval and ((S - n) % progress_interval == 0 or n == 0):
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
                   f"t={t:.6f} jtv_batch_ms={jtv_ms:.2f} w_partial={partial:.4f} "
                   f"elapsed={fmt_time(elapsed)} eta=NA eta_class={fmt_time(eta)}")
            logger.log({
                "phase": "adjoint", "class_global": class_global_idx,
                "class_local": class_local_idx, "backward_step": done,
                "S": S, "t": t, "jtv_batch_ms": jtv_ms, "w_partial": partial,
            }, stdout_msg=msg)

    adj_time = time.time() - adj_t0

    # Per-class means: E over (N, K) -> shape (len(report_indices),).
    class_mean = w_per_traj.mean(dim=(1, 2)).cpu().numpy()                # (T,)
    class_traj_mean = w_per_traj.mean(dim=2).cpu().numpy()                # (T, N) -> transposed below
    # Reorient to (N, T) for the aggregate npz (T = len(report_indices)).
    class_traj_mean = class_traj_mean.T                                   # (N, T)

    # Peak GPU memory (rank-local snapshot).
    peak_mem_gib = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)) \
        if device.type == "cuda" else 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    result = {
        "class_mean":       class_mean.astype(np.float64),        # (T,)
        "class_traj_mean":  class_traj_mean.astype(np.float64),   # (N, T)
        "M_c":              int(M),
        "N":                int(N),
        "K":                int(K),
        "S":                int(S),
        "report_indices":   list(report_indices),
        "fwd_time_sec":     float(fwd_time),
        "adj_time_sec":     float(adj_time),
        "mean_v_batch_ms":  float(np.mean(v_latency)) if v_latency else float("nan"),
        "mean_jtv_batch_ms":float(np.mean(jtv_latency)) if jtv_latency else float("nan"),
        "p50_v_batch_ms":   float(np.median(v_latency)) if v_latency else float("nan"),
        "p90_v_batch_ms":   float(np.quantile(v_latency, 0.9)) if v_latency else float("nan"),
        "p99_v_batch_ms":   float(np.quantile(v_latency, 0.99)) if v_latency else float("nan"),
        "p50_jtv_batch_ms": float(np.median(jtv_latency)) if jtv_latency else float("nan"),
        "p90_jtv_batch_ms": float(np.quantile(jtv_latency, 0.9)) if jtv_latency else float("nan"),
        "p99_jtv_batch_ms": float(np.quantile(jtv_latency, 0.99)) if jtv_latency else float("nan"),
        "peak_gpu_mem_gib": peak_mem_gib,
        "resp_entropy_mean":    float(np.mean(resp_entropy_samples)) if resp_entropy_samples else float("nan"),
        "resp_neff_mean":       float(np.mean(resp_neff_samples)) if resp_neff_samples else float("nan"),
        "resp_sumcheck_max_dev":float(resp_sumcheck_max_dev),
    }
    return result


# ============================================================
# JVP validation (analytic vs central finite difference)
# ============================================================

@torch.no_grad()
def integrate_partial_analytic(
    Z_start: torch.Tensor, Y: torch.Tensor, s_start: int, S: int
) -> torch.Tensor:
    """Continue analytic Euler integration from index s_start to S."""
    if s_start >= S:
        return Z_start
    Z = Z_start
    h = 1.0 / S
    for n in range(s_start, S):
        t = n / S
        v, _, _ = velocity(Z, Y, t)
        Z = Z + h * v
    return Z


@torch.no_grad()
def jvp_validation(
    Y: torch.Tensor,
    class_seed: int,
    N_val: int, K_val: int, S: int,
    solver_index: int,
    eta_list: list,
    device: torch.device,
) -> dict:
    """Compare analytic adjoint  Phi^T q  with central FD JVP.

    Analytic:   propagate q backward from t=1 to t = s_start / S using J^T.
    FD:         forward integrate (Z + delta q) and (Z - delta q) from s_start
                and use  (F(Z+dq) - F(Z-dq)) / (2 delta).

    We compare the *norms* of the analytic backward adjoint vs the FD tangent
    result at t = solver_index / S. If Phi is well-approximated, we expect
        ||F(Z+dq) - F(Z-dq)|| / (2 delta) approximately equals ||Phi q||.
    Analytic ||a_{s_start}|| provides the reference.
    """
    _, D = Y.shape
    h = 1.0 / S

    gen = torch.Generator(device=device).manual_seed(int(class_seed))
    Z0 = torch.randn((N_val, D), device=device, dtype=torch.float32, generator=gen)

    # Forward flow, keeping only the state at solver_index (and full for adjoint).
    Z = Z0
    Z_states = [Z]
    for n in range(S):
        t = n / S
        v, _, _ = velocity(Z, Y, t)
        Z = Z + h * v
        Z_states.append(Z)
    Z_mid = Z_states[solver_index]

    # Draw K_val unit-norm Rademacher probes at t=1.
    q_gen = torch.Generator(device=device).manual_seed(int(class_seed) + 314159)
    r = torch.empty((K_val, N_val, D), device=device, dtype=torch.float32)
    r.bernoulli_(0.5, generator=q_gen).mul_(2.0).sub_(1.0).div_(math.sqrt(D))
    A = r

    for n in range(S - 1, solver_index - 1, -1):
        t = n / S
        alpha, mu = responsibilities(Z_states[n], Y, t)
        JT_A = apply_JT_batched(A, Z_states[n], Y, t, alpha, mu)
        A = A + h * JT_A
    analytic_norm_sq = (A.to(torch.float64) * A.to(torch.float64)).sum(dim=-1)   # (K, N)

    # Reference direction vector for FD: use the SAME probes q (which is r/sqrt(D)).
    q = r                                                                          # (K, N, D)

    # For FD we integrate paired trajectories at each (k, n).
    fd_results = {}
    for eta in eta_list:
        # delta chosen as eta * <local scale>. Use eta * ||Z_mid|| / sqrt(D) per traj
        # so the perturbation has the same per-coord RMS as Z_mid (matches Exp A conv.).
        z_rms = Z_mid.pow(2).mean(dim=1, keepdim=True).sqrt()                      # (N, 1)
        delta = eta * z_rms.unsqueeze(0) * math.sqrt(D)                            # (1, N, 1)
        Zp = Z_mid.unsqueeze(0) + delta * q                                        # (K, N, D)
        Zm = Z_mid.unsqueeze(0) - delta * q
        # Integrate each pair from solver_index to S.
        Zp_final = integrate_partial_analytic(
            Zp.reshape(-1, D), Y, solver_index, S
        ).reshape(K_val, N_val, D)
        Zm_final = integrate_partial_analytic(
            Zm.reshape(-1, D), Y, solver_index, S
        ).reshape(K_val, N_val, D)
        tang = (Zp_final - Zm_final) / (2.0 * delta.squeeze(-1).unsqueeze(-1))     # (K, N, D)
        fd_norm_sq = (tang.to(torch.float64) ** 2).sum(dim=-1)                     # (K, N)
        rel_err = ((fd_norm_sq - analytic_norm_sq).abs() /
                   analytic_norm_sq.clamp_min(1e-30))
        fd_results[eta] = float(rel_err.mean().item())

    return {
        "analytic_wavg_estimate": float(analytic_norm_sq.mean().item()),
        "fd_relative_error_by_eta": fd_results,
        "solver_index": int(solver_index),
        "t": float(solver_index / S),
        "N_val": int(N_val), "K_val": int(K_val),
    }


# ============================================================
# Aggregation, plotting, output writing (rank 0)
# ============================================================

def bootstrap_ci_axis0(values: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05):
    """Percentile bootstrap over the leading axis (classes). Returns (lo, hi)."""
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
    selected_class_ids: list,     # class-name (str) per selected slot
    per_class_mean: np.ndarray,   # (C, T)
    per_class_traj_mean: np.ndarray,  # (C, N, T)
    per_class_meta: list,         # list of dicts (M_c, N, K, S, timings...)
    bootstrap_replicates: int,
    seed: int,
) -> None:
    """Write all required tables, plots, and the master npz."""
    os.makedirs(out_dir, exist_ok=True)
    C, T = per_class_mean.shape

    # -------- global mean + bootstrap CI over classes -----------------------
    global_mean = per_class_mean.mean(axis=0)
    ci_lo, ci_hi = bootstrap_ci_axis0(per_class_mean, bootstrap_replicates, seed)

    # -------- wavg_global.csv -----------------------------------------------
    with open(os.path.join(out_dir, "wavg_global.csv"), "w") as f:
        f.write("t,mean,ci95_low,ci95_high\n")
        for j in range(T):
            f.write(f"{t_grid[j]:.10f},{global_mean[j]:.10e},"
                    f"{ci_lo[j]:.10e},{ci_hi[j]:.10e}\n")

    # -------- wavg_per_class.csv --------------------------------------------
    with open(os.path.join(out_dir, "wavg_per_class.csv"), "w") as f:
        f.write("class_id,class_name," + ",".join(f"t={t_grid[j]:.6f}" for j in range(T)) + "\n")
        for c in range(C):
            row = ",".join(f"{per_class_mean[c, j]:.10e}" for j in range(T))
            f.write(f"{c},{selected_class_ids[c]},{row}\n")

    # -------- class-time heatmaps -------------------------------------------
    plt = _lazy_plt()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(per_class_mean, aspect="auto", origin="lower",
                   extent=[t_grid[0], t_grid[-1], 0, C], cmap="viridis")
    ax.set_xlabel("t")
    ax.set_ylabel("class rank (selection order)")
    ax.set_title("w_avg(t) per class -- selection order")
    plt.colorbar(im, ax=ax, label="w_avg(t)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "wavg_class_time_heatmap_original.png"), dpi=140)
    plt.close()

    # Sort by integrated amplification.
    A_c = np.trapz(per_class_mean, t_grid, axis=1)
    order = np.argsort(A_c)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(per_class_mean[order], aspect="auto", origin="lower",
                   extent=[t_grid[0], t_grid[-1], 0, C], cmap="viridis")
    ax.set_xlabel("t")
    ax.set_ylabel("class rank (sorted by integrated w_avg)")
    ax.set_title("w_avg(t) per class -- sorted by integrated amplification")
    plt.colorbar(im, ax=ax, label="w_avg(t)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "wavg_class_time_heatmap_sorted.png"), dpi=140)
    plt.close()

    # -------- global curves (linear + log-y) --------------------------------
    for scale, fname in [("linear", "wavg_global_linear.png"),
                         ("log",    "wavg_global_logy.png")]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(t_grid, global_mean, marker="o", linewidth=2,
                label="global mean w_avg(t)")
        ax.fill_between(t_grid, ci_lo, ci_hi, alpha=0.25,
                        label="95% class-bootstrap CI")
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="w=1")
        ax.scatter([1.0], [1.0], color="red", zorder=5, label="terminal (t=1)")
        ax.set_xlabel("t")
        ax.set_ylabel("w_avg(t)")
        if scale == "log":
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        ax.set_title(f"Analytic-marginal w_avg(t)   ({scale}-y)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=140)
        plt.close()

    # -------- all-class curves ---------------------------------------------
    for scale, fname in [("linear", "wavg_all_classes_linear.png"),
                         ("log",    "wavg_all_classes_logy.png")]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for c in range(C):
            ax.plot(t_grid, per_class_mean[c], color="steelblue",
                    linewidth=0.5, alpha=0.35)
        ax.plot(t_grid, global_mean, color="black", linewidth=2.2,
                label="global mean")
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel("t")
        ax.set_ylabel("w_avg(t)")
        if scale == "log":
            ax.set_yscale("log")
        ax.set_title(f"w_avg(t) for {C} classes")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=140)
        plt.close()

    # -------- distribution across classes at selected times -----------------
    rep_times = [0.0, 0.25, 0.5, 0.75, 0.9375, 1.0]
    # Nearest report index for each requested time.
    nearest_j = [int(np.argmin(np.abs(t_grid - rt))) for rt in rep_times]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, (rt, j) in enumerate(zip(rep_times, nearest_j)):
        vals = per_class_mean[:, j]
        axes[i].hist(vals, bins=min(20, max(4, C // 2)),
                     color="steelblue", edgecolor="black", alpha=0.8)
        axes[i].axvline(vals.mean(), color="red", linestyle="--",
                        label=f"mean={vals.mean():.3g}")
        axes[i].set_title(f"t≈{rt}  (actual={t_grid[j]:.4f})")
        axes[i].set_xlabel("w_avg,c")
        axes[i].set_ylabel("classes")
        axes[i].grid(alpha=0.3)
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "wavg_class_distribution_selected_times.png"),
                dpi=140)
    plt.close()

    # -------- per-time statistics CSV ---------------------------------------
    with open(os.path.join(out_dir, "wavg_class_statistics_by_time.csv"), "w") as f:
        f.write("t,mean,median,std,iqr,min,max,cv\n")
        for j in range(T):
            v = per_class_mean[:, j]
            f.write(f"{t_grid[j]:.10f},{v.mean():.10e},{np.median(v):.10e},"
                    f"{v.std():.10e},{np.subtract(*np.quantile(v, [0.75, 0.25])):.10e},"
                    f"{v.min():.10e},{v.max():.10e},"
                    f"{(v.std() / (v.mean() + 1e-30)):.10e}\n")

    # -------- per-class summary CSV -----------------------------------------
    with open(os.path.join(out_dir, "wavg_class_summary.csv"), "w") as f:
        f.write("class_id,class_name,integral_A_c,max,argmax_t,"
                "mean_early_t_le_0.25,mean_mid_0.25_0.75,mean_late_gt_0.75,"
                "early_late_ratio,M_c\n")
        for c in range(C):
            v = per_class_mean[c]
            A_int = float(np.trapz(v, t_grid))
            j_max = int(np.argmax(v))
            early_mask = t_grid <= 0.25
            mid_mask   = (t_grid > 0.25) & (t_grid <= 0.75)
            late_mask  = t_grid > 0.75
            m_early = float(v[early_mask].mean()) if early_mask.any() else float("nan")
            m_mid   = float(v[mid_mask].mean())   if mid_mask.any()   else float("nan")
            m_late  = float(v[late_mask].mean())  if late_mask.any()  else float("nan")
            ratio = m_early / (m_late + 1e-30) if late_mask.any() else float("nan")
            f.write(f"{c},{selected_class_ids[c]},{A_int:.10e},"
                    f"{v.max():.10e},{t_grid[j_max]:.10f},"
                    f"{m_early:.10e},{m_mid:.10e},{m_late:.10e},"
                    f"{ratio:.10e},{per_class_meta[c]['M_c']}\n")

    # -------- integral ranking plot -----------------------------------------
    A_int_all = np.array([float(np.trapz(per_class_mean[c], t_grid)) for c in range(C)])
    order_r = np.argsort(A_int_all)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.25 * C)))
    ax.barh(range(C), A_int_all[order_r], color="steelblue")
    ax.set_yticks(range(C))
    ax.set_yticklabels([selected_class_ids[i] for i in order_r], fontsize=7)
    ax.set_xlabel("integrated w_avg(t)")
    ax.set_title("Class ranking by integrated amplification")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "wavg_class_integral_ranking.png"), dpi=140)
    plt.close()

    # -------- uncertainty decomposition -------------------------------------
    # probe variance: within trajectory, across K; trajectory: within class, across N;
    # class: across C. Compute per report time.
    probe_var = np.zeros(T)
    traj_var  = per_class_traj_mean.var(axis=1).mean(axis=0)                  # avg over C
    class_var = per_class_mean.var(axis=0)
    se_class  = per_class_mean.std(axis=0) / max(1.0, math.sqrt(C))

    # We don't have per-probe raw values here (we averaged K probes inside run_class
    # to save memory); traj_var and class_var are what we can decompose exactly.
    with open(os.path.join(out_dir, "wavg_uncertainty_decomposition.csv"), "w") as f:
        f.write("t,var_trajectory_within_class,var_class,se_class\n")
        for j in range(T):
            f.write(f"{t_grid[j]:.10f},{traj_var[j]:.10e},"
                    f"{class_var[j]:.10e},{se_class[j]:.10e}\n")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_grid, traj_var,  marker="o", label="Var_traj_within_class")
    ax.plot(t_grid, class_var, marker="s", label="Var_class")
    ax.plot(t_grid, se_class,  marker="^", label="SE_class")
    ax.set_xlabel("t")
    ax.set_ylabel("variance / SE")
    ax.set_yscale("log")
    ax.set_title("Uncertainty decomposition of w_avg(t)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "wavg_uncertainty_decomposition.png"), dpi=140)
    plt.close()

    # -------- master npz ---------------------------------------------------
    np.savez(
        os.path.join(out_dir, "wavg_raw.npz"),
        times=t_grid,
        selected_class_ids=np.array(selected_class_ids),
        class_mean=per_class_mean,
        class_trajectory_mean=per_class_traj_mean,
        global_mean=global_mean,
        global_ci_lower=ci_lo,
        global_ci_upper=ci_hi,
    )

    # class name mapping
    with open(os.path.join(out_dir, "class_id_to_name.json"), "w") as f:
        json.dump({str(i): selected_class_ids[i] for i in range(C)}, f, indent=2)


# ============================================================
# Microbenchmark
# ============================================================

@torch.no_grad()
def microbenchmark(Y: torch.Tensor, N: int, K: int, S: int, device: torch.device,
                   num_iters: int = 5) -> dict:
    """Time one batched v(z,t) and one J^T A on a representative class."""
    M, D = Y.shape
    Z = torch.randn((N, D), device=device, dtype=torch.float32)
    t = 0.5

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    # warmup
    for _ in range(2):
        _v, alpha, mu = velocity(Z, Y, t)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # v(z, t)
    v_times = []
    for _ in range(num_iters):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.time()
        v, alpha, mu = velocity(Z, Y, t)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        v_times.append((time.time() - t0) * 1000.0)

    # J^T A
    A = torch.randn((K, N, D), device=device, dtype=torch.float32) / math.sqrt(D)
    jtv_times = []
    for _ in range(num_iters):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.time()
        _ = apply_JT_batched(A, Z, Y, t, alpha, mu)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        jtv_times.append((time.time() - t0) * 1000.0)

    return {
        "M_c":       int(M),
        "N":         int(N),
        "K":         int(K),
        "v_ms":      {"median": float(np.median(v_times)),
                      "p90":    float(np.quantile(v_times, 0.9)),
                      "p99":    float(np.quantile(v_times, 0.99))},
        "jtv_ms":    {"median": float(np.median(jtv_times)),
                      "p90":    float(np.quantile(jtv_times, 0.9)),
                      "p99":    float(np.quantile(jtv_times, 0.99))},
        "predicted_forward_sec_per_class":
            float(S * np.median(v_times) / 1000.0),
        "predicted_backward_sec_per_class":
            float(S * np.median(jtv_times) / 1000.0),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="w_avg(t) on the analytic ImageNet marginal flow"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--device", type=str, default=None,
                        help="Single-GPU device (only used without torchrun)")
    cli = parser.parse_args()

    cfg = load_config(cli.config)

    # ---- distributed setup ------------------------------------------------
    default_device = cli.device or cfg.get("device", "cuda:0")
    rank, world_size, device_str = setup_dist(default_device)
    device = torch.device(device_str)

    # ---- output directory -------------------------------------------------
    out_root = cli.output or cfg.get("output_dir") \
        or os.path.join(cfg.get("experiments_base_dir", "./experiment"),
                        cfg["experiment"]["name"])
    if rank == 0:
        os.makedirs(out_root, exist_ok=True)
        os.makedirs(os.path.join(out_root, "per_class"), exist_ok=True)
        shutil.copy(cli.config, os.path.join(out_root, "resolved_config.yaml"))
    barrier(world_size)

    # ---- logger -----------------------------------------------------------
    log_path = os.path.join(out_root, f"progress.rank{rank}.jsonl")
    logger = JsonlLogger(log_path, rank, world_size,
                         echo=bool(cfg.get("logging", {}).get("echo_stdout", True)))

    # ---- reproducibility (FP32 only) --------------------------------------
    seed = int(cfg["experiment"].get("seed", 2026))
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed + rank * 101)

    # ---- data -------------------------------------------------------------
    data_cfg = cfg["data"]
    latent_root = data_cfg["latent_root"]
    num_test_classes = int(data_cfg["num_test_classes"])

    all_classes = list_packed_classes(latent_root)
    if rank == 0:
        print(f"Found {len(all_classes)} classes at {latent_root}")

    # Deterministic class selection.
    rng = np.random.default_rng(seed)
    sel_idx = rng.choice(len(all_classes), size=num_test_classes, replace=False)
    sel_idx = sorted(sel_idx.tolist())
    selected_classes = [all_classes[i] for i in sel_idx]

    if rank == 0:
        with open(os.path.join(out_root, "selected_classes.json"), "w") as f:
            json.dump({"seed": seed,
                       "num_available_classes": len(all_classes),
                       "num_test_classes": num_test_classes,
                       "selected_indices": sel_idx,
                       "selected_class_names": selected_classes}, f, indent=2)

    # Round-robin sharding across ranks.
    local_class_indices = [c for c in range(num_test_classes) if c % world_size == rank]
    if rank == 0:
        print(f"Class sharding: each rank handles ~{len(local_class_indices)} classes.")
    logger.log({
        "phase": "setup",
        "world_size": world_size,
        "num_available_classes": len(all_classes),
        "num_test_classes": num_test_classes,
        "local_class_count": len(local_class_indices),
        "seed": seed,
    })

    # ---- flow / estimator params -----------------------------------------
    S = int(cfg["flow"]["solver_steps"])
    report_indices = list(cfg["flow"]["report_indices"])
    assert 0 in report_indices and S in report_indices, \
        f"report_indices must include 0 and {S}"
    N = int(cfg["estimator"]["samples_per_class"])
    K = int(cfg["estimator"]["probes_per_sample"])
    bootstrap_replicates = int(cfg["estimator"].get("bootstrap_replicates", 10000))
    progress_interval = int(cfg["logging"].get("progress_interval_steps", 8))

    # ---- microbenchmark ---------------------------------------------------
    if rank == 0:
        print("\n[microbenchmark] loading class 0 on rank 0 for latency probing...")
        Y_probe = load_class_latents_fixed(
            latent_root, selected_classes[0], device, seed + sel_idx[0]
        )
        bench = microbenchmark(Y_probe, N=N, K=K, S=S, device=device)
        pred_total_per_class = bench["predicted_forward_sec_per_class"] \
            + bench["predicted_backward_sec_per_class"]
        # Assume ~equal class cost per rank.
        classes_per_rank = math.ceil(num_test_classes / world_size)
        pred_total_run = pred_total_per_class * classes_per_rank
        bench["predicted_total_run_seconds"] = float(pred_total_run)
        bench["classes_per_rank_worst_case"] = int(classes_per_rank)
        with open(os.path.join(out_root, "runtime_profile.json"), "w") as f:
            json.dump(bench, f, indent=2)
        print(f"[microbenchmark] v_ms(p50)={bench['v_ms']['median']:.2f}   "
              f"jtv_ms(p50)={bench['jtv_ms']['median']:.2f}")
        print(f"[microbenchmark] predicted total run ≈ "
              f"{fmt_time(pred_total_run)} (worst rank).")
        del Y_probe
        if device.type == "cuda":
            torch.cuda.empty_cache()
    barrier(world_size)

    # ---- iterate over local classes ---------------------------------------
    per_class_results = {}   # local_key = global class index -> result dict
    resume_completed = set()

    # Support resume: skip classes with an existing checkpoint file.
    for c_global in local_class_indices:
        ckpt_path = os.path.join(out_root, "per_class",
                                 f"class_{c_global:04d}.npz")
        if os.path.exists(ckpt_path):
            resume_completed.add(c_global)

    if resume_completed:
        logger.log({"phase": "resume", "skipped_classes": sorted(resume_completed)})
        if rank == 0:
            print(f"[resume] rank 0 will skip {len(resume_completed)} already-done classes.")

    run_t0 = time.time()

    for local_i, c_global in enumerate(local_class_indices):
        ckpt_path = os.path.join(out_root, "per_class",
                                 f"class_{c_global:04d}.npz")
        if c_global in resume_completed:
            data = np.load(ckpt_path)
            per_class_results[c_global] = {
                "class_mean":      data["class_mean"],
                "class_traj_mean": data["class_traj_mean"],
                "M_c":  int(data["M_c"]),
                "N":    int(data["N"]),
                "K":    int(data["K"]),
                "S":    int(data["S"]),
                "class_name": str(data["class_name"]),
            }
            continue

        cls_name = selected_classes[c_global]
        if rank == 0 or True:
            logger.log({
                "phase": "load_class",
                "class_global": c_global,
                "class_local": local_i + 1,
                "class_name": cls_name,
            }, stdout_msg=(
                f"[rank {rank}] loading class {c_global:03d} ({cls_name}) "
                f"local {local_i + 1}/{len(local_class_indices)}"))

        # Per-class deterministic seed (Q1: fixed epsilon per image, no resampling).
        class_seed = seed * 10007 + sel_idx[c_global]
        Y = load_class_latents_fixed(latent_root, cls_name, device, class_seed)

        cls_t0 = time.time()
        try:
            result = run_class(
                Y=Y,
                class_seed=class_seed + 1,
                N=N, K=K, S=S,
                report_indices=report_indices,
                device=device,
                logger=logger,
                rank=rank,
                class_local_idx=local_i + 1,
                class_local_total=len(local_class_indices),
                class_global_idx=c_global + 1,
                class_global_total=num_test_classes,
                progress_interval=progress_interval,
            )
        except RuntimeError as e:
            logger.log({"phase": "abort", "class_global": c_global,
                        "class_name": cls_name, "error": str(e)},
                       stdout_msg=f"[rank {rank}] ABORT class {c_global}: {e}")
            # Write diagnostic and re-raise so the job stops (safer than silent skip).
            raise

        cls_time = time.time() - cls_t0

        # atomic checkpoint write. np.savez appends '.npz' if the filename
        # doesn't already end with it, so make the tmp path end in '.npz'
        # explicitly to avoid ambiguity.
        tmp_path = ckpt_path + f".tmp.{os.getpid()}.npz"
        np.savez(
            tmp_path,
            class_mean=result["class_mean"],
            class_traj_mean=result["class_traj_mean"],
            M_c=result["M_c"], N=result["N"], K=result["K"], S=result["S"],
            report_indices=np.array(result["report_indices"]),
            class_name=cls_name,
            class_global=c_global,
            fwd_time_sec=result["fwd_time_sec"],
            adj_time_sec=result["adj_time_sec"],
            mean_v_batch_ms=result["mean_v_batch_ms"],
            mean_jtv_batch_ms=result["mean_jtv_batch_ms"],
            peak_gpu_mem_gib=result["peak_gpu_mem_gib"],
            resp_entropy_mean=result["resp_entropy_mean"],
            resp_neff_mean=result["resp_neff_mean"],
            resp_sumcheck_max_dev=result["resp_sumcheck_max_dev"],
        )
        os.replace(tmp_path, ckpt_path)

        result["class_name"] = cls_name
        per_class_results[c_global] = result

        elapsed = time.time() - run_t0
        eta_total = elapsed * (len(local_class_indices) - (local_i + 1)) / max(1, local_i + 1)
        logger.log({
            "phase": "class_done",
            "class_global": c_global,
            "class_local": local_i + 1,
            "class_name": cls_name,
            "M_c": result["M_c"],
            "runtime_sec": round(cls_time, 3),
            "mean_v_batch_ms":  result["mean_v_batch_ms"],
            "mean_jtv_batch_ms":result["mean_jtv_batch_ms"],
            "peak_gpu_mem_gib": result["peak_gpu_mem_gib"],
        }, stdout_msg=(
            f"[rank {rank}] class_done={c_global + 1:03d}/{num_test_classes:02d} "
            f"M_c={result['M_c']} runtime_sec={cls_time:.2f} "
            f"mean_v_batch_ms={result['mean_v_batch_ms']:.2f} "
            f"mean_jtv_batch_ms={result['mean_jtv_batch_ms']:.2f} "
            f"peak_gpu_mem_gib={result['peak_gpu_mem_gib']:.2f} "
            f"elapsed={fmt_time(elapsed)} eta_rank={fmt_time(eta_total)} "
            f"output=per_class/class_{c_global:04d}.npz"
        ))

        del Y
        if device.type == "cuda":
            torch.cuda.empty_cache()

    barrier(world_size)

    # ============================================================
    # Aggregation: rank 0 gathers all per-class checkpoints from disk
    # ============================================================
    if rank == 0:
        print("\n[aggregate] loading per-class checkpoints...")
        T = len(report_indices)
        C = num_test_classes
        per_class_mean = np.zeros((C, T), dtype=np.float64)
        per_class_traj_mean = np.zeros((C, N, T), dtype=np.float64)
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
            per_class_meta[c_global] = {
                "M_c": int(data["M_c"]), "class_name": str(data["class_name"]),
                "fwd_time_sec": float(data["fwd_time_sec"]),
                "adj_time_sec": float(data["adj_time_sec"]),
                "mean_v_batch_ms": float(data["mean_v_batch_ms"]),
                "mean_jtv_batch_ms": float(data["mean_jtv_batch_ms"]),
                "peak_gpu_mem_gib": float(data["peak_gpu_mem_gib"]),
                "resp_entropy_mean": float(data["resp_entropy_mean"]),
                "resp_neff_mean":    float(data["resp_neff_mean"]),
                "resp_sumcheck_max_dev": float(data["resp_sumcheck_max_dev"]),
            }
        if missing:
            raise RuntimeError(
                f"Missing per-class checkpoints for indices: {missing}. "
                f"Re-run to complete these classes (resume is supported)."
            )

        t_grid = np.array([s / S for s in report_indices], dtype=np.float64)
        write_outputs(
            out_dir=out_root,
            t_grid=t_grid,
            selected_class_ids=selected_classes,
            per_class_mean=per_class_mean,
            per_class_traj_mean=per_class_traj_mean,
            per_class_meta=per_class_meta,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )

        # numerical health summary
        max_dev = max(m["resp_sumcheck_max_dev"] for m in per_class_meta if m)
        terminal_w = float(per_class_mean[:, report_indices.index(S)].mean())
        health = {
            "terminal_w_avg_at_t1": terminal_w,
            "terminal_deviation_from_1": abs(terminal_w - 1.0),
            "max_softmax_sum_deviation": max_dev,
            "num_classes": C,
            "any_nan_or_inf": False,
        }
        with open(os.path.join(out_root, "numerical_health.json"), "w") as f:
            json.dump(health, f, indent=2)

        # runtime by class csv
        with open(os.path.join(out_root, "runtime_by_class.csv"), "w") as f:
            f.write("class_id,class_name,M_c,fwd_time_sec,adj_time_sec,"
                    "mean_v_batch_ms,mean_jtv_batch_ms,peak_gpu_mem_gib\n")
            for c in range(C):
                m = per_class_meta[c]
                f.write(f"{c},{m['class_name']},{m['M_c']},"
                        f"{m['fwd_time_sec']:.3f},{m['adj_time_sec']:.3f},"
                        f"{m['mean_v_batch_ms']:.4f},{m['mean_jtv_batch_ms']:.4f},"
                        f"{m['peak_gpu_mem_gib']:.3f}\n")

        # responsibility statistics csv
        with open(os.path.join(out_root, "responsibility_statistics.csv"), "w") as f:
            f.write("class_id,class_name,M_c,resp_entropy_mean,resp_neff_mean,"
                    "resp_sumcheck_max_dev\n")
            for c in range(C):
                m = per_class_meta[c]
                f.write(f"{c},{m['class_name']},{m['M_c']},"
                        f"{m['resp_entropy_mean']:.6e},"
                        f"{m['resp_neff_mean']:.6e},"
                        f"{m['resp_sumcheck_max_dev']:.6e}\n")

        print(f"\n[done] Terminal w_avg(1) = {terminal_w:.6f} (should be ≈ 1)")
        print(f"[done] Outputs in {out_root}")

    barrier(world_size)

    # ============================================================
    # Optional solver-convergence + JVP validation subruns (rank 0 only)
    # ============================================================
    val_cfg = cfg.get("validation", {})
    do_val = bool(val_cfg.get("enabled", True))

    if do_val and rank == 0:
        print("\n[validation] running solver-convergence + JVP checks on rank 0...")
        val_classes = val_cfg.get("num_classes", 8)
        val_class_globals = list(range(min(int(val_classes), num_test_classes)))
        solver_steps_list = list(val_cfg.get("solver_steps", [128, 256, 512]))
        fd_eta_list = list(val_cfg.get("finite_difference_eta", [1.0e-4, 1.0e-3]))

        # ---- 10.7 solver convergence ---------------------------------------
        conv_records = []
        for c_val in val_class_globals:
            cls_name = selected_classes[c_val]
            class_seed = seed * 10007 + sel_idx[c_val]
            Y = load_class_latents_fixed(latent_root, cls_name, device, class_seed)

            # Small N for validation to stay within budget.
            N_val = int(val_cfg.get("val_N", 8))
            K_val = int(val_cfg.get("val_K", 16))

            wavg_by_S = {}
            for S_alt in solver_steps_list:
                # Build a scaled report grid: linearly-spaced 16 points including 0 and S_alt.
                rep_alt = list(np.linspace(0, S_alt, 16, dtype=int).tolist())
                rep_alt[0] = 0
                rep_alt[-1] = S_alt
                res = run_class(
                    Y=Y, class_seed=class_seed + 2,
                    N=N_val, K=K_val, S=S_alt,
                    report_indices=rep_alt,
                    device=device, logger=logger, rank=0,
                    class_local_idx=1, class_local_total=1,
                    class_global_idx=c_val + 1, class_global_total=len(val_class_globals),
                    progress_interval=0,   # quiet
                    resp_stats_every=0,
                    latency_probe_interval=0,
                )
                wavg_by_S[S_alt] = res["class_mean"]

            # relative error vs S_ref (= max solver_steps)
            S_ref = max(solver_steps_list)
            for S_alt in solver_steps_list:
                if S_alt == S_ref:
                    continue
                # Align by fractional t (use nearest indexing).
                t_alt = np.linspace(0, 1, 16)
                t_ref = np.linspace(0, 1, 16)
                rel_err = np.abs(wavg_by_S[S_alt] - wavg_by_S[S_ref]) / (wavg_by_S[S_ref] + 1e-12)
                conv_records.append({
                    "class_id": c_val, "class_name": cls_name,
                    "S": S_alt, "S_ref": S_ref,
                    "max_rel_err": float(rel_err.max()),
                    "mean_rel_err": float(rel_err.mean()),
                })

            # ---- 10.7 & 7.6 JVP validation --------------------------------
            jvp_records = []
            for solver_idx_pct in (0.5,):   # midpoint of the flow
                s_mid = S // 2
                r = jvp_validation(
                    Y=Y, class_seed=class_seed + 3,
                    N_val=N_val, K_val=K_val, S=S,
                    solver_index=s_mid,
                    eta_list=fd_eta_list, device=device,
                )
                r.update({"class_id": c_val, "class_name": cls_name})
                jvp_records.append(r)
            del Y
            torch.cuda.empty_cache() if device.type == "cuda" else None

        # write outputs
        with open(os.path.join(out_root, "solver_convergence.csv"), "w") as f:
            f.write("class_id,class_name,S,S_ref,max_rel_err,mean_rel_err\n")
            for r in conv_records:
                f.write(f"{r['class_id']},{r['class_name']},{r['S']},"
                        f"{r['S_ref']},{r['max_rel_err']:.6e},"
                        f"{r['mean_rel_err']:.6e}\n")

        with open(os.path.join(out_root, "jvp_validation.csv"), "w") as f:
            etas_hdr = ",".join(f"fd_rel_err_eta_{eta:.0e}" for eta in fd_eta_list)
            f.write(f"class_id,class_name,solver_index,t,analytic_wavg,{etas_hdr}\n")
            for r in jvp_records:
                errs = ",".join(f"{r['fd_relative_error_by_eta'][eta]:.6e}"
                                for eta in fd_eta_list)
                f.write(f"{r['class_id']},{r['class_name']},"
                        f"{r['solver_index']},{r['t']:.6f},"
                        f"{r['analytic_wavg_estimate']:.6e},{errs}\n")

        # simple plots
        plt = _lazy_plt()
        if conv_records:
            fig, ax = plt.subplots(figsize=(8, 5))
            for S_alt in solver_steps_list:
                errs = [r["max_rel_err"] for r in conv_records if r["S"] == S_alt]
                if errs:
                    ax.plot(range(len(errs)), errs, marker="o", label=f"S={S_alt}")
            ax.set_yscale("log")
            ax.set_xlabel("validation class")
            ax.set_ylabel("max relative error vs S=%d" % max(solver_steps_list))
            ax.set_title("Solver-convergence relative error")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_root, "wavg_solver_relative_error.png"), dpi=140)
            plt.close()

        print(f"[validation] wrote solver_convergence.csv, jvp_validation.csv")

    barrier(world_size)

    logger.close()
    if world_size > 1:
        dist.destroy_process_group()

    if rank == 0:
        print(f"\nAll done. Results at: {out_root}")


if __name__ == "__main__":
    main()
