# Latent-variance study — path-weighted flow matching vs. data scale

> **Question.** Does the benefit of the parameterised weighting
> `vanilla_weighting_v` depend on the variance of the training latents?
> Two 50k-image ImageNet subsets are built that are **identical except for the
> latent scale** (σ = 0.5 and σ = 2.0, against the ImageNet reference σ ≈ 0.82),
> and four SiT-B/2 models are trained on each.

## TL;DR

```bash
# Stage 1 — datasets (steps 1-2 need no GPU, step 3 uses 8)
./remote_bash_script/build_variance_subsets.sh

# Stage 2 — eight independent 8-GPU runs, launch each on its own node
./remote_bash_script/train_B_velocity_imagenet50k_std05.sh
./remote_bash_script/train_B_vanilla_weighting_05_scale_imagenet50k_std05.sh
./remote_bash_script/train_B_vanilla_weighting_10_scale_imagenet50k_std05.sh
./remote_bash_script/train_B_vanilla_weighting_20_scale_imagenet50k_std05.sh
./remote_bash_script/train_B_velocity_imagenet50k_std20.sh
./remote_bash_script/train_B_vanilla_weighting_05_scale_imagenet50k_std20.sh
./remote_bash_script/train_B_vanilla_weighting_10_scale_imagenet50k_std20.sh
./remote_bash_script/train_B_vanilla_weighting_20_scale_imagenet50k_std20.sh

# Stage 3 — FID/IS vs. training step, one 8-GPU sweep per trained model
./remote_bash_script/run_exp_model_validation_config_B_velocity_imagenet50k_std05.sh
./remote_bash_script/run_exp_model_validation_config_B_vanilla_05_imagenet50k_std05.sh
./remote_bash_script/run_exp_model_validation_config_B_vanilla_10_imagenet50k_std05.sh
./remote_bash_script/run_exp_model_validation_config_B_vanilla_20_imagenet50k_std05.sh
./remote_bash_script/run_exp_model_validation_config_B_velocity_imagenet50k_std20.sh
./remote_bash_script/run_exp_model_validation_config_B_vanilla_05_imagenet50k_std20.sh
./remote_bash_script/run_exp_model_validation_config_B_vanilla_10_imagenet50k_std20.sh
./remote_bash_script/run_exp_model_validation_config_B_vanilla_20_imagenet50k_std20.sh
```

---

## Stage 1 — the two datasets

### Why one base subset, not two

The two subsets are **the same 50,000 images**. Only the multiplicative latent
scale differs, so the variance is the single manipulated variable — anything
else (class balance, image content, encoder noise) is held fixed. Drawing two
independent 50k samples would add sampling noise on top of the effect being
measured.

### Why no re-encoding

The training latent is

```
z = (mean + std · eps) · 0.18215
```

and scaling is linear, so storing `(s·mean, s·std)` produces exactly `s·z`.
The full ImageNet train set is already encoded at
`ILSVRC/latents_packed/train` (79 GB, produced by `encode_dataset.py` +
`pack_latents.py`), so [`build_variance_subset.py`](../build_variance_subset.py)
**slices** the sampled rows out of it and multiplies them. The result is
bit-identical to re-encoding, at zero VAE cost. `--source images` re-runs the
VAE instead, for the case where the packed latents are unavailable.

### The scale

| subset | target σ | scale `s` | predicted σ |
|---|---|---|---|
| `imagenet50k_std0.5` | 0.5 | 0.5 / 0.82 = **0.6097561** | 0.5 |
| `imagenet50k_std2.0` | 2.0 | 2.0 / 0.82 = **2.4390244** | 2.0 |

`σ = sqrt((1/d)·tr Cov(z))` — the same estimator `exp_latent_variance.py`
reports, so 0.82 is directly comparable.

The script **measures** σ on the actual 50k subset and reports it, but derives
the scale from `--source_std 0.82` by default so the numbers match the configs
exactly. A 50k draw typically measures σ ≈ 0.83, i.e. the realised σ lands
around 0.506 / 2.025 rather than exactly 0.5 / 2.0. Pass `--source_std auto`
to hit the targets exactly instead — but then `data.latent_scale` in the eight
configs must be updated to the values printed in `scale_info.json`.

### Sampling

`--sampling stratified` (default) takes 50 images per class, so all 1000
classes are equally represented and the class → label mapping matches the full
ImageNet ordering. `--sampling uniform` draws uniformly over all 1.28M images
instead (≈39 ± 6 per class); ImageNet train is nearly balanced, so the two
agree in expectation, but stratified removes class-count noise from the
comparison.

### Output

```
ILSVRC/variance_subsets/
    subset_manifest.json          # which image is in which row, seed, counts
    base_stats.json               # measured σ of the unscaled subset
    imagenet50k_std0.5/
        scale_info.json           # target/source σ, latent_scale, realised σ
        latents_packed/train/<class>.npy      # (50, 2, 2, 4, 32, 32) float32
        reference_images -> ../reference_images/train
    imagenet50k_std2.0/           # same, scale 2.4390244
    reference_images/train/<class>/<name>.png # FID reference batch
```

≈3.2 GB per latent subset, ≈7 GB for the reference images.

### FID reference images

[`decode_latents.py`](../decode_latents.py) decodes the stored latents back to
images, in one of two modes. **This study uses `scaled`** — each subset gets its
own reference set, and the validation sweep decodes generated latents the same
way (`model_overrides.latent_scale: 1.0`).

| mode | decodes | reference set |
|---|---|---|
| `scaled` (used here) | `z_stored / 0.18215` — scale left in | one per subset, different pixel distributions |
| `unscaled` (default) | `z_stored / (0.18215 · s)` — true reconstruction | identical for both subsets |

Measured on a 3-image probe of each subset:

| subset | mean pixel | pixel std | pixels clipped at 0/255 |
|---|---|---|---|
| σ=0.5 | 124.2 | 55.8 | 0.0 % |
| σ=2.0 | 131.7 | 104.3 | **47.6 %** |

Both sides of each FID pass through the same map, so the metric is internally
consistent and the four weightings within a dataset are directly comparable.
Two consequences to keep in mind: FID **magnitudes do not transfer across the
two subsets** (different pixel spaces), and at σ=2.0 nearly half the reference
pixels are saturated to pure black/white, which compresses the dynamic range
Inception sees and may blunt FID's sensitivity there.

Switching to true-colour reconstructions is a one-line change: set
`model_overrides.latent_scale: null` in the validation configs (the true scale
is then read from the checkpoint) and re-decode with `--decode_mode unscaled`
into a single shared reference folder. Under that mode both subsets decode to
byte-identical images — verified end-to-end, max pixel difference 0.

`--make_npz` additionally writes an ADM-style `.npz`; it holds all images in
memory first (~10 GB at 50k × 256²), so leave it off unless needed.

---

## Stage 2 — the eight training runs

| dataset | weighting | `loss_space` | λ | `scale_enable` | config / script suffix |
|---|---|---|---|---|---|
| σ=0.5 | baseline | `velocity` | – | false | `velocity_imagenet50k_std05` |
| σ=0.5 | vanilla | `vanilla_weighting_v` | 0.5 | true | `vanilla_weighting_05_scale_imagenet50k_std05` |
| σ=0.5 | vanilla | `vanilla_weighting_v` | 1.0 | true | `vanilla_weighting_10_scale_imagenet50k_std05` |
| σ=0.5 | vanilla | `vanilla_weighting_v` | 2.0 | true | `vanilla_weighting_20_scale_imagenet50k_std05` |
| σ=2.0 | (same four) | | | | `..._imagenet50k_std20` |

Everything else is shared: SiT-B/2, 256², batch 1024, seed 42, Linear path,
velocity prediction, `cfg_scale 1.0`, from scratch (`ckpt: null`), 8 GPUs each.

### 200k steps

`training.max_train_steps: 200000` (new, see below) is the stopping criterion.
50,000 images at batch 1024 with `drop_last=True` gives **48 steps/epoch**, so
`epochs: 4200` is only an upper bound that guarantees the step budget is
reachable (4200 × 48 = 201,600).

### Experiment names

`train.py` builds the run directory from the config, and `data.dataset_name`
feeds its suffix, so the name tracks the dataset automatically:

```
053-SiT-B-2-scaled-Linear-velocity-None-vanilla_weighting_v-lam1.0-IS256-BS1024-imagenet50k-std0.5
054-SiT-B-2-Linear-velocity-None-velocity-IS256-BS1024-imagenet50k-std2.0
```

All eight names are distinct. The numeric prefix is
`len(glob(results/*))` at start-up, so **runs launched at the same instant can
share a prefix** — harmless here (the rest of the name disambiguates), but do
not use the prefix alone to identify a run.

### Disk

8 runs × 8 checkpoints × ≈2.1 GB (model + EMA + optimiser, SiT-B) ≈ **133 GB**,
plus ≈14 GB of datasets. `/scratch` was at 1.4 TB free when this was written —
enough, but worth watching. Lower `ckpt_every` at your own risk.

---

## Stage 3 — validation sweeps

One `exp_model_validation.py` config + launch script per trained model,
`configs/exp_model_validation_config_B_{velocity,vanilla_05,vanilla_10,vanilla_20}_imagenet50k_std{05,20}.yaml`.
Each evaluates FID and IS at

```
steps = [25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000]
```

matching `ckpt_every: 25000`, with 50k generated images, 50-step Euler ODE,
`cfg_scale 1.0`, EMA weights, seed 42 — the same protocol as the existing
SiT-B ImageNet sweeps, so the curves are directly comparable in shape.
Results land in `experiment/performance_SiT_B_<weighting>_imagenet50k_std{05,20}/`.

### Reference batch — each subset uses its own

`ref_data_path` points at `<subset>/reference_images/train`, and
`model_overrides.latent_scale: 1.0` makes the sweep decode generated latents
with plain `z / 0.18215`, i.e. **without** dividing the dataset scale out. Both
sides of the FID therefore live in the same pixel space as the model's own
training data.

The scale still reaches the sweep automatically when you want it: `train.py`
saves `args` (including `latent_scale`) in every checkpoint, and
`resolve_model_args` reads it. The explicit `1.0` override is what suppresses
it. See the reference-batch table in Stage 1 for what this costs in FID
interpretability.

### Checkpoint paths

`checkpoint_dir` is written **without** the `NNN-` index that `train.py`
prepends (it is not knowable before the run starts).
`exp_model_validation.py` now resolves `results/<run-name>/checkpoints` against
`results/*<run-name>/checkpoints` and errors out if the match is not unique, so
no manual editing is needed after training. All eight paths were verified to
equal the names `train.py` builds from the eight training configs.

- **`train.py`**
  - `training.max_train_steps` (optional, default `null`) — hard stop on
    optimisation steps, checked after the checkpoint/sampling blocks so a final
    step landing on a multiple of `ckpt_every` is still saved. Every rank
    evaluates the same `train_steps`, so all ranks break together and no
    collective operation is left dangling.
  - `data.latent_scale` (optional, default `1.0`) — the factor baked into the
    stored latents. The dataset on disk is already scaled, so training reads it
    unchanged; the value is used to **divide the scale out before VAE decoding**
    (W&B sample grids and the validation FID) and to apply it when latents are
    encoded on the fly from raw images.
- **`sample_ddp.py`** — `--latent-scale` (default 1.0), same division before
  decoding. Pass `0.6097561` / `2.4390244` when sampling from these runs;
  forgetting it makes every generated image wrong and the FID meaningless.
- **`exp_model_validation.py`**
  - The decode is now `samples / (0.18215 * latent_scale)`. `latent_scale` is
    read from `ckpt["args"]` (which `train.py` saves), so the validation configs
    need no manual value; `model_overrides.latent_scale` forces one if needed.
    Checkpoints predating the field resolve to `1.0`, i.e. unchanged behaviour.
  - `resolve_checkpoint_dir()` tolerates the `NNN-` run-directory prefix.
- Both defaults are no-ops, so all existing configs and scripts behave exactly
  as before (verified against `sit_config_B_velocity_imagenet_400k.yaml`).

## Caveats worth knowing before reading the results

- **The loss magnitude is not comparable across the two datasets.** The
  velocity target scales with the data, so the σ=2.0 loss is ≈16× the σ=0.5
  loss at equal learning rate. With a fixed lr of 1e-4 (hard-coded in
  `train.py`) the σ=2.0 runs therefore take effectively larger steps. That is
  inherent to changing the data variance without renormalising; compare
  *weightings within a dataset* first, and treat cross-dataset comparisons as
  confounded by this unless the learning rate is swept.
- `scale_enable: true` + `extra_scale: 0.845` for the vanilla runs and
  `false` + `1.0` for the baseline follow the existing SiT-B ImageNet series
  (see [05_new_weightings.md](05_new_weightings.md),
  [06_new_weighting_experiments.md](06_new_weighting_experiments.md)) so the new
  numbers line up with the ones already collected.
- 50k images at batch 1024 for 200k steps is ≈4100 passes over the data — these
  runs are firmly in the memorisation regime, which is intended for a variance
  probe but is not comparable to the 1.28M-image runs.
- `data.data_path` still points at the full ImageNet train directory as the
  raw-image fallback. If `packed_latent_data_path` is ever mistyped, training
  silently falls back to **all 1.28M images** instead of the 50k subset; check
  the `Dataset contains 50,000 images` line in the log at start-up.
