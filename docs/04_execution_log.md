# Execution Log — 2026-07-22

This document records the actual outcome of the cleanup session.

## User decisions (from chat)

1. **TM2T removal** → ✅ full removal.
2. **Folder rename fixes** → only fix `SiT` → `Path-Weighted_Flow_Matching`.
   Other stale absolute paths (dataset paths on old server,
   `Generative_Rendering` sys.path traversal) are **left untouched** because
   the project will still be run on the old server.
3. **Experiment cleanup** → prune everything **except** `min_snr` and the
   `velocity` / `vanilla_weighting` / `straight_weighting` variants.

## Files deleted

### TM2T (9 files)
- Python: `train_tm2t.py`, `sample_tm2t.py`, `eval_tm_sweep.py`,
  `plot_tm_sweep.py`
- Shell:  `train_tm2t.sh`, `eval_tm_sweep.sh`, `eval_tm_sweep_2.sh`,
  `eval_tm_sweep_temp.sh`
- Config: `configs/tm2t_config.yaml`

### Top-level shell scripts (3 files)
- `eval_alpha_sweep_temp.sh`
- `run_alpha_sweep.sh`
- `train_vt.sh`

### Configs (50 files)
- 5 × `eval_alpha_sweep_*.yaml`
- 20 × `sit_config_*_blend_*.yaml` (constant/linear × xv/xn × entire/scale)
- 7 × `sit_config_{B,L,S,XL}_constant_blend_xv_entire_*.yaml`
- 13 × `sit_config_{,B_,L_,S_,XL_}vv*.yaml`, `sit_config_vt*.yaml`,
       `sit_config_vn*.yaml`, `sit_config_nn.yaml`
- 9 × `exp_model_validation_config_*.yaml` (blend / vt / vn / vv / nn / tt /
       linear_blend variants)

### Remote bash scripts (46 files)
- All `run_alpha_sweep_*.sh`
- All `train_*blend*.sh` (per-size + generic)
- All `train_*vv*.sh`, `train_vt*.sh`, `train_vn*.sh`, `train_nn.sh`
- All `run_validation_*_blend_*.sh`, `run_validation_vt.sh`

### `min_snr` preserved ✅
- `configs/sit_config_min_snr_scale.yaml`
- `configs/sit_config_XL_vanilla_weighting_10_scale_min_snr_imagenet.yaml`
- `configs/exp_model_validation_config_min_snr.yaml`
- `configs/exp_model_validation_config_XL_minsnr_imagenet.yaml`
- `remote_bash_script/train_min_snr_scale.sh`
- `remote_bash_script/train_XL_vanilla_weighting_10_scale_min_snr_imagenet.sh`
- `remote_bash_script/run_exp_model_validation_config_XL_minsnr_imagenet.sh`

## Files modified

### `remote_bash_script/*.sh` (bulk rename)
- Replaced `cd /scratch/project/prj-02-visual-ai/hkzhang/SiT` with
  `cd /scratch/project/prj-02-visual-ai/hkzhang/Path-Weighted_Flow_Matching`
  in every remaining shell script.
- 122 files now contain the new path (survivors of the deletion pass).
- Note: `conda activate SiT` was **left as-is** — it refers to the conda
  environment name, not the folder.

### `transport/transport.py` (comments only, no logic change)
- Line 137: comment about `t_min` no longer mentions TM2T.
- Line 193: comment inside `sample()` no longer mentions TM2T.
- Line 640: `sample_ode` docstring no longer mentions TM2T; still explains
  `t0_override` / `t1_override` for general two-model composition use.

### `CLAUDE.md`
- Rewrote "What This Is" section to describe path-weighted flow matching
  focus instead of TM2T.
- Removed the "Training TM2T" and "Transport mode sweep evaluation"
  command blocks.
- Removed `train_tm2t.py` from the core-files list.
- Replaced the "TM2T-specific config additions" YAML block with a
  "Parameterized-weighting config example" block.

## Verification checks performed

- ✅ `python3 -c "import ast; ast.parse(open('transport/transport.py').read())"`
  → syntax OK.
- ✅ `grep -r "tm2t\|TM2T"` → matches only inside `docs/` (planning history).
- ✅ `grep -r "hkzhang/SiT"` → matches only inside `docs/` (planning history).
- ✅ No shell script or config references a deleted file.

## Follow-up deletions (resolved 2026-07-22)

The user confirmed deletion of all three flagged orphans:

- ✅ **`eval_alpha_sweep.py`** — deleted (all its configs and launchers were
  already removed, so this was an orphan).
- ✅ **`run_SiT.ipynb`** — deleted (upstream demo notebook, no longer needed).
- ✅ **`train.log`, `tea_debug.log`** — deleted (runtime artifacts).

Note: the `run_SiT.ipynb` links inside `README.md` still resolve — they
point to the upstream `github.com/willisma/SiT` repo (external URLs), not
to the local file.

## Remaining kept items

- **`test_fid_is.py`** — standalone test script, independent of removed
  experiments. Kept as-is.
- **`README.md`** — upstream README preserved; external GitHub / Colab
  links unchanged.
- **`visuals/`** — sample visualization images from the upstream repo.
- **`LICENSE.txt`, `environment.yml`, `CONFIG_GUIDE.md`** — untouched.
