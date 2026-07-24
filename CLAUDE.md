# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Coding Requirement from Instructions

0. When you execute any following instructions, you need to say which instruction you will execute next in the chat box. If no following instruction is executed, say you do not need to execute them in chat box.

1. After each large update (like add a function, or change the working logic), need to execute the instruction: "Check if this change introduce any logical error. If so, fix it."

2. After each update related to variable type transform, or change the type of variable compared to orginal version. Check if you made any mistake.

## What This Is

SiT (Scalable Interpolant Transformers) is a flow/diffusion-based generative image model built on the DiT backbone, trained on class-conditional ImageNet. It extends DiT with a flexible interpolant framework supporting multiple path types (Linear, GVP, VP) and model predictions (velocity, score, noise).

This repo focuses on **path-weighted flow matching**: comparing the baseline (plain velocity MSE) against **parameterized weightings** (`vanilla_weighting_v`, `straight_weighting_v`) that re-weight the per-timestep loss by an analytical formula parameterized by `λ` (`loss_lambda`). Other loss-space variants (blends, min_snr, cross-loss) are implemented in `transport/transport.py` but are not exercised by the current experiment scripts.

## Commands

**Training SiT:**
```bash
./train.sh
# or directly:
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=<N> --master_port=<PORT> train.py --config configs/sit_config.yaml
```

**Sampling (single GPU):**
```bash
python sample.py ODE --image-size 256 --seed 1
python sample.py ODE --model SiT-L/4 --image-size 256 --ckpt /path/to/model.pt
```

**Sampling (multi-GPU, for FID evaluation):**
```bash
torchrun --nproc_per_node=N sample_ddp.py ODE --model SiT-XL/2 --num-fid-samples 50000
```

**FID computation:** Uses `FID.py` with ADM-compatible `.npz` output from `sample_ddp.py`.

## Architecture

### Core files
- [models.py](models.py) — `SiT` transformer model (DiT backbone). Uses `timm`'s `PatchEmbed`, `Attention`, `Mlp`. Conditions on timestep via `TimestepEmbedder` (sinusoidal → MLP) and class label via `LabelEmbedder` (with CFG dropout). AdaLN-zero modulation in each block.
- [train.py](train.py) — standard single-stage SiT training with DDP, EMA, periodic sampling/FID, wandb.
- [train_utils.py](train_utils.py) — sampler argument parsing, ODE/SDE config helpers.
- [wandb_utils.py](wandb_utils.py) — W&B logging helpers.

### Transport module (`transport/`)
- `transport.py` — `Transport` class: wraps path sampler + model type, computes training loss, builds ODE/SDE samplers.
- `path.py` — interpolant path definitions: `ICPlan` (Linear), `GVPCPlan` (GVP), `VPCPlan` (VP). Each defines `alpha_t`, `sigma_t` and their derivatives.
- `integrators.py` — ODE (via `torchdiffeq`) and SDE (Euler/Heun) integrators.
- `utils.py` — `EasyDict`, logging helpers.

### Data flow
Raw images → VAE encoder (Stable Diffusion `AutoencoderKL`, 8× spatial compression) → latents → SiT transformer → velocity/score/noise prediction → transport loss. Sampling reverses the ODE/SDE in latent space → VAE decoder → image.

## Configuration System

All training parameters live in YAML configs under `configs/`. A `--config` argument is **required** — there are no CLI overrides. Key sections: `data`, `model`, `transport`, `training`, `logging`, `validation`, `sampling`, `checkpoint`.

**Parameterized-weighting config example:**
```yaml
transport:
  path_type: 'Linear'
  prediction: 'velocity'
  loss_space: 'vanilla_weighting_v'   # or 'straight_weighting_v'
  loss_lambda: 1.0                    # λ hyper-parameter
  scale_loss: true                    # optional: normalize weight magnitude
```

Model variants: `SiT-XL/2` (675M), `SiT-L/2` (458M), `SiT-B/2` (130M), `SiT-S/2` (33M). The `/2` suffix is patch size.

## W&B Logging

W&B credentials can be set in the config file directly (`wandb_key`, `wandb_entity`, `wandb_project`) or via environment variables `WANDB_KEY`, `ENTITY`, `PROJECT`. The config values take precedence over env vars.
