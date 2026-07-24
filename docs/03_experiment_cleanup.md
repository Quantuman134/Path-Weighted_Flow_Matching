# Experiment Cleanup Plan

Date: 2026-07-22
Status: **Executed.**

**User decision:** keep `min_snr` variants alongside `velocity` (baseline)
and `vanilla_weighting_v` / `straight_weighting_v` (parameterized). Prune
everything else (blends, vv/vt/vn/nn/tt, alpha-sweep).

## Scope

The user now only cares about the comparison:

- **Baseline** → `prediction: velocity` + unweighted MSE
  (files matching `*velocity*`)
- **Parameterized weighting** → `prediction: velocity` +
  `loss_space: vanilla_weighting_v` / `straight_weighting_v`
  (files matching `*vanilla*` / `*straight*`)

The loss-space *implementations* in
[`transport/transport.py`](../transport/transport.py) stay untouched (per user
directive) — only **launch scripts, config files, and experiment YAMLs** for
irrelevant comparisons are removed.

## Categories to remove

### A. All `*_blend_*` (path-blend variants)
- `constant_blend_xv`, `linear_blend_xv`, `constant_blend_xn`,
  `linear_blend_xn`, `constant_blend_vn`, `linear_blend_vn`, and their
  `_entire` and `_scale` variants.

### B. All `*_vv`, `*_vt`, `*_vn`, `*_nn`, `*_tt`
- Cross-prediction / cross-loss-space experiments (velocity-target,
  velocity-noise, noise-noise, target-target, etc.).

### C. ~~`min_snr` variants~~ — **KEPT** (user request)

### D. Alpha-sweep experiments
- Everything with `alpha_sweep` in the name — these compare *two-model
  velocity blends*, orthogonal to the baseline-vs-parameterized study.

### E. TM-sweep experiments (handled in [`01_tm2t_removal.md`](01_tm2t_removal.md))

## Files to delete

### 5.1 Config files (`configs/*.yaml`)

**Alpha sweep configs (all delete):**
```
configs/eval_alpha_sweep_config_vn.yaml
configs/eval_alpha_sweep_config_xn.yaml
configs/eval_alpha_sweep_config_xv.yaml
configs/eval_alpha_sweep_target_noise_config.yaml
configs/eval_alpha_sweep_target_velocity_config.yaml
```

**Blend / cross-loss training configs (all delete):**
```
configs/sit_config_constant_blend_xn.yaml
configs/sit_config_constant_blend_xn_scale.yaml
configs/sit_config_constant_blend_xn_entire.yaml
configs/sit_config_constant_blend_xn_entire_scale.yaml
configs/sit_config_constant_blend_xv.yaml
configs/sit_config_constant_blend_xv_scale.yaml
configs/sit_config_constant_blend_xv_entire.yaml
configs/sit_config_constant_blend_xv_entire_scale.yaml
configs/sit_config_linear_blend_xn.yaml
configs/sit_config_linear_blend_xn_scale.yaml
configs/sit_config_linear_blend_xn_entire.yaml
configs/sit_config_linear_blend_xn_entire_scale.yaml
configs/sit_config_linear_blend_xv.yaml
configs/sit_config_linear_blend_xv_scale.yaml
configs/sit_config_linear_blend_xv_entire.yaml
configs/sit_config_linear_blend_xv_entire_scale.yaml

configs/sit_config_B_constant_blend_xv_entire_cifar10.yaml
configs/sit_config_B_constant_blend_xv_entire_scale_cifar10.yaml
configs/sit_config_L_constant_blend_xv_entire_scale_cifar10.yaml
configs/sit_config_L_constant_blend_xv_entire_scale_imagenet.yaml
configs/sit_config_S_constant_blend_xv_entire_cifar10.yaml
configs/sit_config_S_constant_blend_xv_entire_scale_cifar10.yaml
configs/sit_config_XL_constant_blend_xv_entire_scale_cifar10.yaml

configs/sit_config_vv.yaml
configs/sit_config_vt.yaml
configs/sit_config_vt_scale.yaml
configs/sit_config_vn_scale.yaml
configs/sit_config_nn.yaml
configs/sit_config_B_vv_cifar10.yaml
configs/sit_config_B_vv_imagenet.yaml
configs/sit_config_L_vv_cifar10.yaml
configs/sit_config_L_vv_imagenet.yaml
configs/sit_config_S_vv_cifar10.yaml
configs/sit_config_S_vv_imagenet.yaml
configs/sit_config_XL_vv_cifar10.yaml
configs/sit_config_XL_vv_imagenet.yaml
configs/sit_config_XL_vv_imagenet_fintuning.yaml

```

(min_snr configs kept per user request: `sit_config_min_snr_scale.yaml`,
`sit_config_XL_vanilla_weighting_10_scale_min_snr_imagenet.yaml`.)

**Validation configs for irrelevant experiments:**
```
configs/exp_model_validation_config_constant_blend_xn.yaml
configs/exp_model_validation_config_constant_blend_xv.yaml
configs/exp_model_validation_config_linear_blend_xv.yaml
configs/exp_model_validation_config_linear_blend_xv_entire.yaml
configs/exp_model_validation_config_nn.yaml
configs/exp_model_validation_config_tt.yaml
configs/exp_model_validation_config_vn.yaml
configs/exp_model_validation_config_vt.yaml
configs/exp_model_validation_config_vv.yaml
```

(min_snr validation configs kept: `exp_model_validation_config_min_snr.yaml`,
`exp_model_validation_config_XL_minsnr_imagenet.yaml`.)

### 5.2 Shell scripts (top-level)
```
eval_alpha_sweep_temp.sh
run_alpha_sweep.sh
```

### 5.3 Shell scripts (`remote_bash_script/`)

**Alpha-sweep runners:**
```
remote_bash_script/run_alpha_sweep_vn.sh
remote_bash_script/run_alpha_sweep_xn.sh
remote_bash_script/run_alpha_sweep_xv.sh
```

**Blend / cross-loss trainers:**
```
remote_bash_script/train_constant_blend_xn.sh
remote_bash_script/train_constant_blend_xn_scale.sh
remote_bash_script/train_constant_blend_xn_entire.sh
remote_bash_script/train_constant_blend_xn_entire_scale.sh
remote_bash_script/train_constant_blend_xv.sh
remote_bash_script/train_constant_blend_xv_scale.sh
remote_bash_script/train_constant_blend_xv_entire.sh
remote_bash_script/train_constant_blend_xv_entire_scale.sh
remote_bash_script/train_linear_blend_xn.sh
remote_bash_script/train_linear_blend_xn_scale.sh
remote_bash_script/train_linear_blend_xn_entire.sh
remote_bash_script/train_linear_blend_xn_entire_scale.sh
remote_bash_script/train_linear_blend_xv.sh
remote_bash_script/train_linear_blend_xv_scale.sh
remote_bash_script/train_linear_blend_xv_entire.sh
remote_bash_script/train_linear_blend_xv_entire_scale.sh

remote_bash_script/train_B_constant_blend_xv_entire_cifar10.sh
remote_bash_script/train_B_constant_blend_xv_entire_scale_cifar10.sh
remote_bash_script/train_L_constant_blend_xv_entire_scale_cifar10.sh
remote_bash_script/train_L_constant_blend_xv_entire_scale_imagenet.sh
remote_bash_script/train_S_constant_blend_xv_entire_cifar10.sh
remote_bash_script/train_S_constant_blend_xv_entire_scale_cifar10.sh
remote_bash_script/train_XL_constant_blend_xv_entire_scale_cifar10.sh

remote_bash_script/train_vv.sh
remote_bash_script/train_vv copy.sh
remote_bash_script/train_vt.sh
remote_bash_script/train_vt_scale.sh
remote_bash_script/train_vn_scale.sh
remote_bash_script/train_nn.sh
remote_bash_script/train_S_vv.sh
remote_bash_script/train_S_vv_cifar10.sh
remote_bash_script/train_S_vv_imagenet.sh
remote_bash_script/train_B_vv_cifar10.sh
remote_bash_script/train_B_vv_imagenet.sh
remote_bash_script/train_L_vv_cifar10.sh
remote_bash_script/train_L_vv_imagenet.sh
remote_bash_script/train_XL_vv_cifar10.sh
remote_bash_script/train_XL_vv_imagenet.sh
remote_bash_script/train_XL_vv_imagenet_finetuning.sh

```

(min_snr shell scripts kept: `remote_bash_script/train_min_snr_scale.sh`,
`remote_bash_script/train_XL_vanilla_weighting_10_scale_min_snr_imagenet.sh`,
`remote_bash_script/run_exp_model_validation_config_XL_minsnr_imagenet.sh`.)

**Validation runners for irrelevant experiments:**
```
remote_bash_script/run_validation_constant_blend_xn.sh
remote_bash_script/run_validation_constant_blend_xv.sh
remote_bash_script/run_validation_linear_blend_xv.sh
remote_bash_script/run_validation_linear_blend_xv_entire.sh
remote_bash_script/run_validation_vt.sh
```

### 5.4 Top-level launcher
```
train_vt.sh                       # only invokes sit_config_vt.yaml (deleted)
```

## Files to keep

### Kept configs (baseline vs parameterized only)

**Baseline (velocity) configs:**
```
configs/sit_config.yaml                     # canonical template
configs/imagenet256_example.yaml            # example template
configs/cifar10_config.yaml                 # dataset template
configs/exp_model_validation_config.yaml    # base validation template
configs/exp_velocity_rmse_config.yaml       # velocity RMSE probe

configs/exp_model_validation_config_B_velocity_cifar10.yaml
configs/exp_model_validation_config_B_velocity_imagenet.yaml
configs/exp_model_validation_config_L_velocity_cifar10.yaml
configs/exp_model_validation_config_S_velocity_cifar10.yaml
configs/exp_model_validation_config_S_velocity_imagenet.yaml
configs/exp_model_validation_config_XL_velocity_cifar10.yaml
configs/exp_model_validation_config_XL_velocity_imagenet.yaml
configs/exp_model_validation_config_XL_velocity_imagenet_github.yaml
```

**Parameterized (vanilla / straight) configs:**
```
configs/sit_config_B_vanilla_weighting_05_cifar10.yaml
configs/sit_config_B_vanilla_weighting_05_scale_cifar10.yaml
configs/sit_config_B_vanilla_weighting_05_scale_imagenet.yaml
configs/sit_config_B_vanilla_weighting_10_cifar10.yaml
configs/sit_config_B_vanilla_weighting_10_scale_cifar10.yaml
configs/sit_config_B_vanilla_weighting_10_scale_imagenet.yaml
configs/sit_config_B_vanilla_weighting_20_cifar10.yaml
configs/sit_config_B_vanilla_weighting_20_scale_cifar10.yaml
configs/sit_config_B_vanilla_weighting_20_scale_imagenet.yaml
configs/sit_config_B_straight_weighting_05_cifar10.yaml
configs/sit_config_B_straight_weighting_05_scale_cifar10.yaml
configs/sit_config_B_straight_weighting_10_cifar10.yaml
configs/sit_config_B_straight_weighting_10_scale_cifar10.yaml
configs/sit_config_B_straight_weighting_20_cifar10.yaml
configs/sit_config_B_straight_weighting_20_scale_cifar10.yaml

configs/sit_config_L_vanilla_weighting_05_scale_cifar10.yaml
configs/sit_config_L_vanilla_weighting_05_scale_imagenet.yaml
configs/sit_config_L_vanilla_weighting_10_scale_cifar10.yaml
configs/sit_config_L_vanilla_weighting_10_scale_imagenet.yaml
configs/sit_config_L_vanilla_weighting_20_scale_cifar10.yaml
configs/sit_config_L_vanilla_weighting_20_scale_imagenet.yaml
configs/sit_config_L_straight_weighting_05_scale_imagenet.yaml
configs/sit_config_L_straight_weighting_10_scale_imagenet.yaml
configs/sit_config_L_straight_weighting_20_scale_imagenet.yaml

configs/sit_config_S_vanilla_weighting_05_cifar10.yaml
configs/sit_config_S_vanilla_weighting_05_scale_cifar10.yaml
configs/sit_config_S_vanilla_weighting_05_scale_imagenet.yaml
configs/sit_config_S_vanilla_weighting_10_cifar10.yaml
configs/sit_config_S_vanilla_weighting_10_scale_cifar10.yaml
configs/sit_config_S_vanilla_weighting_10_scale_imagenet.yaml
configs/sit_config_S_vanilla_weighting_20_cifar10.yaml
configs/sit_config_S_vanilla_weighting_20_scale_cifar10.yaml
configs/sit_config_S_vanilla_weighting_20_scale_imagenet.yaml
configs/sit_config_S_straight_weighting_05_cifar10.yaml
configs/sit_config_S_straight_weighting_05_scale_cifar10.yaml
configs/sit_config_S_straight_weighting_10_cifar10.yaml
configs/sit_config_S_straight_weighting_10_scale_cifar10.yaml
configs/sit_config_S_straight_weighting_20_cifar10.yaml
configs/sit_config_S_straight_weighting_20_scale_cifar10.yaml

configs/sit_config_XL_vanilla_weighting_05_scale_cifar10.yaml
configs/sit_config_XL_vanilla_weighting_05_scale_imagenet.yaml
configs/sit_config_XL_vanilla_weighting_05_scale_imagenet_finetuning.yaml
configs/sit_config_XL_vanilla_weighting_10_scale_cifar10.yaml
configs/sit_config_XL_vanilla_weighting_10_scale_imagenet.yaml
configs/sit_config_XL_vanilla_weighting_10_scale_imagenet_finetuning.yaml
configs/sit_config_XL_vanilla_weighting_20_scale_cifar10.yaml
configs/sit_config_XL_vanilla_weighting_20_scale_imagenet.yaml
configs/sit_config_XL_vanilla_weighting_20_scale_imagenet_finetuning.yaml
configs/sit_config_XL_straight_weighting_10_scale_imagenet_finetuning.yaml

configs/exp_model_validation_config_B_vanilla_05_cifar10.yaml
configs/exp_model_validation_config_B_vanilla_10_imagenet.yaml
configs/exp_model_validation_config_B_vanilla_20_imagenet.yaml
configs/exp_model_validation_config_L_vanilla_05_cifar10.yaml
configs/exp_model_validation_config_L_vanilla_05_imagenet.yaml
configs/exp_model_validation_config_L_vanilla_10_cifar10.yaml
configs/exp_model_validation_config_L_vanilla_10_imagenet.yaml
configs/exp_model_validation_config_L_vanilla_20_cifar10.yaml
configs/exp_model_validation_config_L_vanilla_20_imagenet.yaml
configs/exp_model_validation_config_L_straight_20_imagenet.yaml
configs/exp_model_validation_config_S_vanilla_05_cifar10.yaml
configs/exp_model_validation_config_S_vanilla_10_cifar10.yaml
configs/exp_model_validation_config_S_vanilla_10_imagenet.yaml
configs/exp_model_validation_config_XL_vanilla_05_cifar10.yaml
configs/exp_model_validation_config_XL_vanilla_10_imagenet.yaml
configs/exp_model_validation_config_XL_vanilla_10_imagenet_finetuning.yaml
```

### Kept launcher scripts
```
train.sh                             # single-config trainer
run_validation.sh                    # validation driver
run_validation_vanilla_velocity.sh   # main baseline vs vanilla sweep

remote_bash_script/train_S_vanilla_weighting_*
remote_bash_script/train_B_vanilla_weighting_*    (plus *_straight_weighting_*)
remote_bash_script/train_L_vanilla_weighting_*    (plus *_straight_weighting_*)
remote_bash_script/train_XL_vanilla_weighting_*   (plus *_straight_weighting_*)
remote_bash_script/run_exp_model_validation_*_vanilla_*
remote_bash_script/run_exp_model_validation_*_velocity_*
```

## Grand total

- 5 alpha-sweep configs deleted
- ~40 blend / vv / vt / vn / nn / tt / min_snr configs deleted
- 11 irrelevant validation configs deleted
- 2 top-level shell scripts deleted (`eval_alpha_sweep_temp.sh`,
  `run_alpha_sweep.sh`, plus `train_vt.sh`)
- ~45 `remote_bash_script/*.sh` files deleted

**Total ≈ 100+ files deleted; the loss-space code in `transport/` is
completely untouched.**
