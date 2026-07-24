# Folder-rename Path Fixes

Date: 2026-07-22
Status: **Executed.**

**User decision:** the project will still be run on the old server, so
absolute paths that point *outside* the project folder (e.g. dataset paths,
`Generative_Rendering` sys.path insertion) are **still valid** and must be
left untouched. Only fix references that changed due to the `SiT →
Path-Weighted_Flow_Matching` folder rename.

## Background

The project folder was renamed **`SiT` → `Path-Weighted_Flow_Matching`**.

## Audit result

I searched for hard-coded absolute paths and `sys.path` manipulations that
could break after the rename. Findings:

### 1. Broken — remote bash scripts (`cd /scratch/.../SiT`)

Every launcher script under
[`remote_bash_script/`](../remote_bash_script) starts with:

```bash
cd /scratch/project/prj-02-visual-ai/hkzhang/SiT
```

This path no longer exists after the rename. It must become:

```bash
cd /scratch/project/prj-02-visual-ai/hkzhang/Path-Weighted_Flow_Matching
```

**Affected files (44 total, one line each):**

Every `.sh` file under `remote_bash_script/` that contains an *uncommented*
`cd /scratch/.../SiT` on line 4. The single exception is
`remote_bash_script/train_S_vv.sh` where the `cd` is already commented out.

The blanket fix is a single find-and-replace across the folder.

### 2. Safe — Python `sys.path` insertion

Files like [`eval_alpha_sweep.py`](../eval_alpha_sweep.py),
[`eval_tm_sweep.py`](../eval_tm_sweep.py),
[`exp_model_validation.py`](../exp_model_validation.py),
[`exp_velocity_rmse.py`](../exp_velocity_rmse.py),
[`test_fid_is.py`](../test_fid_is.py) all use:

```python
_SIT_DIR = os.path.dirname(os.path.abspath(__file__))
```

This is **folder-name-agnostic** — it derives the path from `__file__` at
runtime and still works. Only the *variable name* `_SIT_DIR` is stylistically
outdated; renaming it is cosmetic and optional.

### 3. Stale but harmless — `FID.py`

[`FID.py`](../FID.py) lines 6-9:

```python
current_dir = os.path.dirname(os.path.abspath(__file__))
generative_rendering_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if generative_rendering_dir not in sys.path:
    sys.path.insert(0, generative_rendering_dir)
```

This walks three levels up looking for a `Generative_Rendering` root, which
matches the old server layout `/home/hkzhang/Generative_Rendering/DiT/T2I_lab/SiT/`.
The only import that would have needed it (`from utils import ...`) is
**already commented out**, so this is dead code. Safe to delete for
cleanliness.

### 4. Broken — dataset path in `configs/cifar10_config.yaml`

```yaml
data_path: /home/hkzhang/Generative_Rendering/DiT/T2I_lab/results/cifar10/train
val_data_path: /home/hkzhang/Generative_Rendering/DiT/T2I_lab/results/cifar10/test
```

These absolute paths point to the old server tree. They are dataset paths
(unrelated to the folder rename per se) — the user should update them to
wherever CIFAR-10 lives now, or we leave them and let the user fix on demand.
**Proposed action: leave as-is with a `TODO` comment** — we can't know the
new dataset location.

### 5. Log files — informational only

[`train.log`](../train.log) and [`tea_debug.log`](../tea_debug.log) contain
old paths but are runtime artifacts. **No action needed**.

### 6. README — external repository refs

[`README.md`](../README.md) contains `github.com/willisma/SiT` — these are
correct external references (the upstream repo), unchanged. **No action
needed**.

## Applied fixes

### Fix A ✅ — remote_bash_script/*.sh

Replace, in every `.sh`:

```bash
cd /scratch/project/prj-02-visual-ai/hkzhang/SiT
```

with

```bash
cd /scratch/project/prj-02-visual-ai/hkzhang/Path-Weighted_Flow_Matching
```

### Fix B ❌ — FID.py: NOT applied (user request)

The `Generative_Rendering` sys.path insertion in `FID.py` lines 3-9 is
left untouched. On the old server the traversal still resolves correctly
(the code will just no-op since no import uses it).

### Fix C ❌ — cifar10_config.yaml: NOT applied (user request)

Dataset paths `/home/hkzhang/Generative_Rendering/DiT/T2I_lab/results/cifar10/...`
are still valid on the old server. Left untouched.

### Fix D — not applied

Cosmetic rename of `_SIT_DIR` → `_PROJECT_DIR` was not applied. The
variable is folder-name-agnostic via `__file__`, so this is a naming
nitpick only.
