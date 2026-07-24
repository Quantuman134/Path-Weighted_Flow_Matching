# Project Overview — Path-Weighted Flow Matching

> This document is the starting point for anyone (human or agent) picking up this
> repository. Read this first, then consult the other files in `docs/`.

## Origin

- Repository forked / based on the official
  [willisma/SiT](https://github.com/willisma/SiT) repository — see
  [`README.md`](../README.md) for the upstream paper (Ma et al.,
  "Exploring Flow and Diffusion-based Generative Models with Scalable
  Interpolant Transformers", NeurIPS 2024).
- The folder was **renamed from `SiT` to `Path-Weighted_Flow_Matching`** to
  reflect the current research focus.

## Current research focus

The primary research question of *this* repository is:

> **How do different weightings `w(t)` over the interpolant path affect
> training and sample quality of a flow-matching model?**

Concretely we compare:

1. **Baseline** — plain velocity-prediction loss (`prediction: velocity`,
   `loss_space: velocity`, unweighted MSE).
2. **Parameterized weighting** — the `vanilla_weighting_v` /
   `straight_weighting_v` loss spaces in
   [`transport/transport.py`](../transport/transport.py), which introduce a
   scalar hyper-parameter `λ` (`loss_lambda` in configs) and re-weight the
   per-timestep MSE according to an analytical formula.

Other loss spaces (blends, `min_snr`, `vt`, `vn`, `nn`, `tt`, …) are still
**implemented** in `transport/transport.py` because the analysis code
references those enum values, but their **experiment / launch scripts have
been pruned** — we no longer run them.

## What has been removed (2026-07-22 cleanup)

- **TM2T** (two-stage model where a frozen pretrained SiT handles
  `t ∈ [0, tₘ]` and a new model handles `t ∈ [tₘ, 1]`) — no longer relevant.
  See [`01_tm2t_removal.md`](01_tm2t_removal.md) for the full removal list.
- Irrelevant path-weighting experiment scripts (blends, alpha sweeps, TM
  sweeps, min_snr, vt/vn/nn/tt, etc.) — see
  [`03_experiment_cleanup.md`](03_experiment_cleanup.md).

## What still lives here

### Core model
- [`models.py`](../models.py) — SiT (DiT-backbone) transformer, unchanged.

### Transport / loss
- [`transport/transport.py`](../transport/transport.py) — `Transport` class,
  `LossSpace` enum (all variants preserved, including the *unused-in-scripts*
  ones), `Sampler`.
- [`transport/path.py`](../transport/path.py) — `ICPlan` (Linear), `GVPCPlan`,
  `VPCPlan`.
- [`transport/integrators.py`](../transport/integrators.py) — ODE/SDE
  integrators.

### Training / sampling
- [`train.py`](../train.py) — single-stage SiT training with DDP, EMA,
  W&B, periodic sampling.
- [`sample.py`](../sample.py), [`sample_ddp.py`](../sample_ddp.py) —
  single- and multi-GPU sampling.
- [`train_utils.py`](../train_utils.py) — sampler arg-parsing helpers.

### Evaluation (kept)
- [`exp_model_validation.py`](../exp_model_validation.py) — the workhorse
  FID/IS evaluation driver for baseline vs. parameterized-weighting
  experiments. Driven by `configs/exp_model_validation_config_*_velocity_*.yaml`
  and `configs/exp_model_validation_config_*_vanilla_*.yaml`.
- [`FID.py`](../FID.py), [`IS.py`](../IS.py) — metric implementations.
- [`exp_velocity_rmse.py`](../exp_velocity_rmse.py) — velocity RMSE probe.

### Data pipeline
- [`encode_dataset.py`](../encode_dataset.py) — pre-encode images to VAE
  latents.
- [`pack_latents.py`](../pack_latents.py) — pack latents for training.
- [`download.py`](../download.py) — pretrained-checkpoint downloader.

### Launch scripts (kept)
- [`train.sh`](../train.sh) — baseline / vanilla training launcher.
- [`run_validation.sh`](../run_validation.sh) — validation launcher.
- [`run_validation_vanilla_velocity.sh`](../run_validation_vanilla_velocity.sh) —
  the full sweep over `{S, B, L, XL}` × `{velocity, vanilla(λ=0.5, 1.0, 2.0)}`
  × `{CIFAR-10, ImageNet}`.
- [`remote_bash_script/`](../remote_bash_script) — per-config train launchers
  (only `velocity` and `vanilla_weighting_*` variants kept).

## Configuration system

All training / evaluation parameters live in YAML files under
[`configs/`](../configs). A `--config` flag is **required** — no CLI
overrides. See [`CONFIG_GUIDE.md`](../CONFIG_GUIDE.md) for the schema.

## Key files to read next

1. [`01_tm2t_removal.md`](01_tm2t_removal.md) — what was removed and why.
2. [`02_folder_rename_fixes.md`](02_folder_rename_fixes.md) — hard-coded paths
   patched after the `SiT → Path-Weighted_Flow_Matching` rename.
3. [`03_experiment_cleanup.md`](03_experiment_cleanup.md) — pruned experiment
   scripts and configs.
4. [`04_execution_log.md`](04_execution_log.md) — actual outcome + open
   questions from the 2026-07-22 cleanup session.
