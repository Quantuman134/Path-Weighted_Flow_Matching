# Change log: Min-SNR removal + new mean-one weightings (2026-07-24)

## Goal
Remove the dedicated **min-SNR** weighting and add the 6 baseline mean-one
weightings listed in `docs/baseline_weighting.html` (Uniform Flow Matching
excluded — it is the vanilla `velocity` baseline already in code).

## 1. Removed: `MIN_SNR` loss space
- `transport/transport.py`: removed `MIN_SNR` enum member + docstring line,
  its `training_losses` branch, its `VELOCITY_LOSS_SCALES` entry, and the
  `min_snr` flag plumbing (`Transport.__init__` param + `self.min_snr`).
- `transport/__init__.py`: removed the `loss_space == "min_snr"` → `MIN_SNR`
  mapping and the `min_snr` parameter in `create_transport`.
- `train.py`: removed `args.min_snr` config parsing, the `min_snr=` passthrough
  in `create_transport(...)`, and the `-minsnr` experiment-name suffix block.
- Configs deleted (all reference the removed `min_snr` loss space / flag):
  - `configs/sit_config_min_snr_scale.yaml`
  - `configs/sit_config_XL_vanilla_weighting_10_scale_min_snr_imagenet.yaml`
  - `configs/exp_model_validation_config_min_snr.yaml`
  - `configs/exp_model_validation_config_XL_minsnr_imagenet.yaml`
  - `remote_bash_script/train_min_snr_scale.sh`
  - `remote_bash_script/train_XL_vanilla_weighting_10_scale_min_snr_imagenet.sh`
- `configs/sit_config.yaml`: dropped `min_snr:` line; updated `loss_space`
  comment to list the new spaces.
- `run_validation.sh`: removed the commented `exp_model_validation_config_min_snr.yaml` line.

> Note: `MIN_SNR_GAMMA_V` is a **different** formula (Hang et al. Min-SNR-γ,
> parameterized by `loss_lambda`) and was *added*, not removed.

## 2. Added: 6 mean-one velocity weightings
All applied directly to the velocity residual: `mean_flat(weight * (model_output - ut)**2)`.
Each got an enum member, a docstring line, a `loss_space` string mapping in
`__init__.py`, a `training_losses` branch, and a `VELOCITY_LOSS_SCALES` entry (1.0,
since they are already mean-one).

| `loss_space` string | Weight `w(t)` | Source | Param |
|---|---|---|---|
| `snr_v` | `3 t^2` | Gagneux et al. | — |
| `kg_v` | `(1/Z)·2t(1-t)/((1-t)^2+e^{-2}t^2)`, Z≈1.373260287 | Kingma & Gao | — |
| `min_snr_gamma_v` | `3(1+√g)^2/g · min(t^2, g(1-t)^2)` | Hang et al. | `g=loss_lambda` (default 5) |
| `logit_normal_v` | `exp(-logit(t)^2/2)/(√(2π)·t(1-t))` (m=0,s=1) | SD3 | — |
| `cosmap_v` | `2/(π(1-2t+2t^2))` | SD3 | — |
| `rfpp_v` | `2 cosh(4(t-1/2))/sinh(4)` | Lee et al. RF++ | — |

`min_snr_gamma_v` was added to `_LAMBDA_LOSS_SPACES` in `train.py`, so its
experiment name appends `-lam{loss_lambda}` (default `-lam5`).

## 3. Experiment-name behavior (answer to user question)
`train.py` builds `experiment_name` from `str(args.loss_space)` (the YAML
string). So setting `loss_space: "snr_v"` automatically yields a folder/run
name containing `snr_v`. **No extra code is needed** — names update
automatically. Only the λ-parameterized spaces (`vanilla_weighting_v`,
`straight_weighting_v`, `min_snr_gamma_v`) append `-lam{λ}`.

## Verification
- `ast.parse` OK for `transport.py`, `transport/__init__.py`, `train.py`.
- No live `loss_space: 'min_snr'` remains; `MIN_SNR` is gone from the enum;
  all 6 new string→enum mappings present.
- Could NOT run a full import (no `torch` in this env); logic verified
  structurally.
