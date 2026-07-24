# TM2T Removal Plan

Date: 2026-07-22
Status: **Proposed — waiting for user confirmation before executing deletions.**

## What is TM2T?

TM2T (Two-stage Model → Two) is a research extension in which:

- A **frozen pretrained SiT** handles the early denoising phase `t ∈ [0, tₘ]`.
- A new **trainable model** handles the later phase `t ∈ [tₘ, 1]`.

This is *no longer part of the current research agenda* and can be fully
removed.

## Files to be deleted

### Python entry points (TM2T-only)
| File | Role |
| --- | --- |
| [`train_tm2t.py`](../train_tm2t.py) | TM2T two-stage training script |
| [`sample_tm2t.py`](../sample_tm2t.py) | Two-stage sampler (frozen SiT → TM2T model) |
| [`eval_tm_sweep.py`](../eval_tm_sweep.py) | Sweep over the `tm` handoff timestep |
| [`plot_tm_sweep.py`](../plot_tm_sweep.py) | Plots `FID vs tm` from the sweep output |

### Shell launchers (TM2T-only)
| File | Role |
| --- | --- |
| [`train_tm2t.sh`](../train_tm2t.sh) | `torchrun train_tm2t.py` wrapper |
| [`eval_tm_sweep.sh`](../eval_tm_sweep.sh) | Runs `eval_tm_sweep.py` over `v2t` configs 0–9 |
| [`eval_tm_sweep_2.sh`](../eval_tm_sweep_2.sh) | Same, over `t2v` configs 0–9 |
| [`eval_tm_sweep_temp.sh`](../eval_tm_sweep_temp.sh) | One-off `eval_tm_sweep` launch |

### Config files (TM2T-only)
| File | Role |
| --- | --- |
| [`configs/tm2t_config.yaml`](../configs/tm2t_config.yaml) | Only TM2T training config |

> Note: the `eval_tm_sweep_{v2t,t2v}_config_{0..9}.yaml` files referenced by
> `eval_tm_sweep.sh` / `eval_tm_sweep_2.sh` do **not** currently exist in
> `configs/` (they were presumably generated on the fly or lived elsewhere).
> There is nothing extra to delete there.

## Code touch-ups (non-deletion, in kept files)

These references become dead but do not need to be scrubbed for correctness.
They **can be left as-is** because the `t_min` kwarg defaults to `0.0`
(a no-op). The plan is:

- **[`transport/transport.py`](../transport/transport.py)** — keep the `t_min`
  parameter (harmless default `0.0`) but rewrite the docstring comments that
  reference TM2T (lines 137, 193, 640–641) to remove TM2T language.
- **[`transport/__init__.py`](../transport/__init__.py)** — leave `t_min=0.0`
  in `create_transport` (used by many non-TM2T callers).
- **[`exp_model_validation.py`](../exp_model_validation.py)** and
  **[`eval_tm_sweep.py`](../eval_tm_sweep.py)** — the second will be deleted,
  the first passes `t_min=0.0` which is fine.

- **[`CLAUDE.md`](../CLAUDE.md)** — remove the TM2T section and the
  `train_tm2t.py`/`eval_tm_sweep.py` references.

## Grand total

- 4 Python files deleted
- 4 shell files deleted
- 1 YAML config deleted
- 2 docs updated (`CLAUDE.md`, `transport/transport.py` comments)
