# New weightings: training & evaluation configs and scripts

Date: 2026-07-24 (revised same day)

> **Revision (2026-07-24):** disabled `scale_enable` (was `true` → now `false`),
> set `extra_scale: 1.0` (was `0.845`), and bumped both `ckpt_every` and
> `sample_every` to `10000` (was `5000`). The 12 validation configs were
> updated in lockstep — their `checkpoint_dir` no longer contains `-scaled-`,
> because `train.py` only adds that suffix when `scale_enable` is true.

## Summary

Generated **48 files** (12 experiments × 4 files each) to train and
evaluate every new mean-one weighting on the **B** and **XL** SiT
architectures with **ImageNet 256**.

## What was generated

For each of the 6 new weightings × 2 model sizes:

| # | weighting `loss_space` | filename tag | uses λ? |
|---|---|---|---|
| 1 | `snr_v`             | `snr_v`                 | no |
| 2 | `kg_v`              | `kg_v`                  | no |
| 3 | `min_snr_gamma_v`   | `min_snr_gamma_v_lam5`  | **yes**, γ = `loss_lambda = 5.0` |
| 4 | `logit_normal_v`    | `logit_normal_v`        | no |
| 5 | `cosmap_v`          | `cosmap_v`              | no |
| 6 | `rfpp_v`            | `rfpp_v`                | no |

For each `(model_size, weighting)` pair the following four files were created:

1. **Training config** → `configs/sit_config_{M}_{tag}_scale_imagenet.yaml`
2. **Training bash script** → `remote_bash_script/train_{M}_{tag}_scale_imagenet.sh`
3. **Validation config** → `configs/exp_model_validation_config_{M}_{tag}_imagenet.yaml`
4. **Validation bash script** → `remote_bash_script/run_exp_model_validation_config_{M}_{tag}_imagenet.sh`

Where `{M} ∈ {B, XL}` and `{tag}` is the filename tag from the table above.

## Reference templates used

- Training config template →
  [`configs/sit_config_B_vanilla_weighting_10_scale_imagenet.yaml`](../configs/sit_config_B_vanilla_weighting_10_scale_imagenet.yaml)
- Training bash template →
  [`remote_bash_script/train_B_vanilla_weighting_10_scale_imagenet.sh`](../remote_bash_script/train_B_vanilla_weighting_10_scale_imagenet.sh)
- Validation config template →
  [`configs/exp_model_validation_config_B_vanilla_10_imagenet.yaml`](../configs/exp_model_validation_config_B_vanilla_10_imagenet.yaml)
- Validation bash template →
  [`remote_bash_script/run_exp_model_validation_config_B_vanilla_10_imagenet.sh`](../remote_bash_script/run_exp_model_validation_config_B_vanilla_10_imagenet.sh)

## Key design decisions

- **`scale_enable: false` + `extra_scale: 1.0`** — no post-hoc loss scaling.
  Every new weighting is already **mean-one** by construction (their
  `VELOCITY_LOSS_SCALES[...] = 1.0` entries reflect this), so the empirical
  correction factor `extra_scale = 0.845` used by the vanilla baselines is
  neither needed nor appropriate. `extra_scale: 1.0` is a no-op when
  `scale_enable: false` and is set explicitly for clarity.
- **`global_batch_size: 1024`**, 3000 epochs, seed 42, 20 workers — identical
  to the existing baselines.
- **`ckpt_every: 10000` + `sample_every: 10000`** — checkpoint & W&B sampling
  cadence is halved to `10000` steps (from `5000`) to reduce disk & sample-
  generation overhead over the full 200k-step run.
- **`ckpt: null`** — every new experiment trains **from scratch**. Resuming
  from a `vanilla_weighting_v` checkpoint would bias the comparison.
- **Checkpoint schedule** for validation:
  - B  → `[25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000]`
  - XL → `[20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000]`
  (Same cadences as `exp_model_validation_config_B_vanilla_10_imagenet.yaml`
  and `exp_model_validation_config_XL_velocity_imagenet.yaml`. All chosen
  steps are multiples of `ckpt_every = 10000`.)
- **`checkpoint_dir`** — set to
  `./results/{ModelString}-Linear-velocity-None-{loss_space}[-lam{λ}]-IS256-BS1024-imagenet/checkpoints/`
  (matching what `train.py` produces — note **no `-scaled` segment**, since
  `scale_enable: false`). The runtime experiment index prefix `NNN-`
  assigned by `train.py` is **omitted** in these templates — after training
  you either rename the results dir to strip the index, or prepend it to
  `checkpoint_dir` before running the validation script.

## How to run

Training a single experiment (e.g. B + SNR weighting):
```bash
bash remote_bash_script/train_B_snr_v_scale_imagenet.sh
```

Once checkpoints are written, evaluate FID/IS across the checkpoint
schedule:
```bash
bash remote_bash_script/run_exp_model_validation_config_B_snr_v_imagenet.sh
```

## Verification

- All 48 files written successfully by the generator.
- Spot-checked: `SiT-XL/2` used for XL configs, `loss_lambda: 5.0` set only
  for `min_snr_gamma_v`, `checkpoint_dir` correctly includes `-lam5.0`
  suffix for `min_snr_gamma_v` and omits it for the others.
- Bash scripts are marked executable (`chmod 755`).
