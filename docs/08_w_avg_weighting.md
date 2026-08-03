# 08 — `w_avg` loss weighting (Food-101, SiT-XL/2)

## The weighting

```
w_avg(t) = 0.3231 + 2.7003 · exp(-3.9094 t)
```

This is the exponential fit of the finite-difference estimate of the endpoint-error
amplification `w_avg(t) = E_{z_t}[(1/d) tr(Φ(1,t)^T Φ(1,t))]` produced by
[exp_w_avg_finite_difference.py](../exp_w_avg_finite_difference.py) (mode `w_avg`,
Experiment B). It is monotonically decreasing: `w(0)=3.0234`, `w(0.5)=0.7055`,
`w(1)=0.3772` — i.e. it up-weights the noisy end of the path, where an error in the
velocity is amplified most by the remaining flow.

**Mean-one:** `∫₀¹ w_avg(t) dt = 0.3231 + 2.7003·(1-e^{-3.9094})/3.9094 = 0.99997`,
so it is already normalized under `t ~ U(0,1)` and needs no post-hoc rescaling
(`scale_enable: false`, like the other mean-one weightings in `05_new_weightings.md`).

## Code changes

Same five registration points as every other velocity weighting:

| File | Change |
|---|---|
| [transport/transport.py](../transport/transport.py) | `LossSpace.W_AVG` enum member + docstring line |
| [transport/transport.py](../transport/transport.py) | module constants `W_AVG_C=0.3231`, `W_AVG_A=2.7003`, `W_AVG_B=3.9094` |
| [transport/transport.py](../transport/transport.py) | `VELOCITY_LOSS_SCALES[LossSpace.W_AVG] = 1.0` |
| [transport/transport.py](../transport/transport.py) | `training_losses` branch: `weight = C + A·exp(-B·t)`, applied to the velocity residual `mean_flat(weight * (model_output - ut)**2)` |
| [transport/__init__.py](../transport/__init__.py) | string mapping `loss_space: 'w_avg'` → `LossSpace.W_AVG` |

`exp_model_validation.py` needs no change: it builds the transport for *sampling*
only (`path_type` / `prediction`), and the loss space does not affect the ODE.

`train.py` needs no change either — the run directory name is built from the
`loss_space` string, so the run lands in
`results/NNN-SiT-XL-2-Linear-velocity-None-w_avg-IS256-BS1024-food101/`.
`w_avg` is *not* in `_LAMBDA_LOSS_SPACES`, so no `-lam…` suffix is appended.

## Experiment files (Food-101 256, SiT-XL/2)

| Purpose | File |
|---|---|
| Training config | [configs/sit_config_XL_w_avg_food101.yaml](../configs/sit_config_XL_w_avg_food101.yaml) |
| Validation config | [configs/exp_model_validation_config_XL_w_avg_food101.yaml](../configs/exp_model_validation_config_XL_w_avg_food101.yaml) |
| Training launcher | [remote_bash_script/train_XL_w_avg_food101.sh](../remote_bash_script/train_XL_w_avg_food101.sh) |
| Validation launcher | [remote_bash_script/run_exp_model_validation_config_XL_w_avg_food101.sh](../remote_bash_script/run_exp_model_validation_config_XL_w_avg_food101.sh) |

Settings mirror the existing Food-101 XL runs so the curves are comparable:
BS 1024, 200,000 steps (`max_train_steps`), seed 42, ckpt/sample every 10,000;
validation over steps 20k…200k, EMA weights, 50,000 samples, cfg 1.0, 50-step
Euler, FID/IS reference = 50,000 images from the Food-101 *train* split.

Run:

```bash
bash remote_bash_script/train_XL_w_avg_food101.sh                       # 8-GPU training
bash remote_bash_script/run_exp_model_validation_config_XL_w_avg_food101.sh   # FID/IS vs steps
```

## Verification performed

- `create_transport(..., loss_space='w_avg')` → `LossSpace.W_AVG`, `scale_loss=False`.
- Mean-one confirmed analytically (0.999969) and by 2M-sample Monte Carlo (0.999846).
- `training_losses` output matches a hand-computed `mean_flat(w(t)·(v̂-u_t)²)` to 1.2e-7.
- Per-sample ratio `loss_w_avg / loss_velocity` equals `w(t)` exactly.
- Both YAMLs parse via `train.py:config_to_args`; the validation `checkpoint_dir`
  string matches the run name `train.py` will generate; all data paths exist on disk.
