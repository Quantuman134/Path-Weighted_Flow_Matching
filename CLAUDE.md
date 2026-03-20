# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Coding Requirement from Instructions

0. When you execute any following instructions, you need to say which instruction you will execute next in the chat box. If no following instruction is executed, say you do not need to execute them in chat box.

1. After each large update (like add a function, or change the working logic), need to execute the instruction: "Check if this change introduce any logical error. If so, fix it."

2. After each update related to variable type transform, or change the type of variable compared to orginal version. Check if you made any mistake.

## What This Is

SiT (Scalable Interpolant Transformers) is a flow/diffusion-based generative image model built on the DiT backbone, trained on class-conditional ImageNet. It extends DiT with a flexible interpolant framework supporting multiple path types (Linear, GVP, VP) and model predictions (velocity, score, noise).

This repo also implements **TM2T** (Two-stage Model), a research extension where a frozen pretrained SiT handles the early denoising phase `[0, t_min]` and a new trainable model handles the later phase `[t_min, 1]`.

## Commands

**Training SiT:**
```bash
./train.sh
# or directly:
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=<N> --master_port=<PORT> train.py --config configs/sit_config.yaml
```

**Training TM2T:**
```bash
./train_tm2t.sh
# or directly:
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=<N> --master_port=<PORT> train_tm2t.py --config configs/tm2t_config.yaml
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

**Transport mode sweep evaluation:**
```bash
./eval_tm_sweep.sh   # runs v2t configs 0–9
python eval_tm_sweep.py --config ./configs/eval_tm_sweep_config.yaml
```

**FID computation:** Uses `FID.py` with ADM-compatible `.npz` output from `sample_ddp.py`.

## Architecture

### Core files
- [models.py](models.py) — `SiT` transformer model (DiT backbone). Uses `timm`'s `PatchEmbed`, `Attention`, `Mlp`. Conditions on timestep via `TimestepEmbedder` (sinusoidal → MLP) and class label via `LabelEmbedder` (with CFG dropout). AdaLN-zero modulation in each block.
- [train.py](train.py) — standard single-stage SiT training with DDP, EMA, periodic sampling/FID, wandb.
- [train_tm2t.py](train_tm2t.py) — TM2T two-stage training. Loads pretrained SiT (frozen) for `t ∈ [0, t_min]`; trains new model for `t ∈ [t_min, 1]`.
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

**TM2T-specific config additions:**
```yaml
transport:
  prediction: 'target'               # TM2T model's prediction head
  pretrained_prediction: 'velocity'  # must match the frozen SiT's checkpoint
tm2t:
  t_min: 0.3                         # split point between stage 1 and stage 2
  pretrained_ckpt: './results/.../checkpoints/XXXXXX.pt'
```

Model variants: `SiT-XL/2` (675M), `SiT-L/2` (458M), `SiT-B/2` (130M), `SiT-S/2` (33M). The `/2` suffix is patch size.

## W&B Logging

W&B credentials can be set in the config file directly (`wandb_key`, `wandb_entity`, `wandb_project`) or via environment variables `WANDB_KEY`, `ENTITY`, `PROJECT`. The config values take precedence over env vars.
