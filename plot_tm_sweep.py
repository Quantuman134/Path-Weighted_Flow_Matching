"""
Plot FID-vs-tm curves for two groups of experiments, showing mean ± std.

Groups
------
  v2t  (velocity → target) : experiment/tm_sweep_velocity_target_{21..30}
  t2v  (target → velocity) : experiment/tm_sweep_target_velocity_{0..8}

Each folder contains fid_results.json  { "0.0": fid, "0.1": fid, ... }
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

# ─── Experiment folder definitions ───────────────────────────────────────────

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment")

GROUPS = {
    "v2t (velocity→target)": [
        os.path.join(BASE_DIR, f"tm_sweep_velocity_target_{i}")
        for i in range(21, 31)          # 21 … 30
    ],
    "t2v (target→velocity)": [
        os.path.join(BASE_DIR, f"tm_sweep_target_velocity_{i}")
        for i in range(0, 9)            # 0 … 8
    ],
}

JSON_FILE = "fid_results.json"
SAVE_PATH = os.path.join(BASE_DIR, "tm_sweep_comparison.png")


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_group(folders: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Load fid_results.json from each folder.
    Returns:
        tms   : sorted array of tm values  (T,)
        matrix: FID matrix                 (N, T)  N = number of runs
    """
    runs = []
    for folder in folders:
        path = os.path.join(folder, JSON_FILE)
        if not os.path.isfile(path):
            print(f"  [warn] missing {path}, skipping")
            continue
        with open(path) as f:
            data = json.load(f)
        # sort by float key
        sorted_items = sorted(data.items(), key=lambda kv: float(kv[0]))
        runs.append([v for _, v in sorted_items])

    tms = np.array([float(k) for k, _ in sorted_items])
    matrix = np.array(runs)          # (N, T)
    print(f"  Loaded {len(runs)} runs, {len(tms)} tm values each")
    return tms, matrix


# ─── Plotting ─────────────────────────────────────────────────────────────────

COLORS = {
    "v2t (velocity→target)": "#2196F3",   # blue
    "t2v (target→velocity)": "#F44336",   # red
}

def plot_groups(groups_data: dict):
    fig, ax = plt.subplots(figsize=(9, 5))

    for label, (tms, matrix) in groups_data.items():
        mean = matrix.mean(axis=0)
        std  = matrix.std(axis=0)
        color = COLORS.get(label, None)

        ax.errorbar(tms, mean, yerr=std, marker="o", linewidth=2,
                    capsize=4, capthick=1.5, elinewidth=1.5,
                    label=label, color=color)

    ax.set_xlabel("tm  (handoff timestep)", fontsize=12)
    ax.set_ylabel("FID", fontsize=12)
    ax.set_title("FID vs Handoff Timestep  —  mean ± std across runs", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.35)
    ax.set_xticks(tms)

    fig.tight_layout()
    fig.savefig(SAVE_PATH, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot → {SAVE_PATH}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    groups_data = {}
    for label, folders in GROUPS.items():
        print(f"\nLoading {label} ...")
        tms, matrix = load_group(folders)
        groups_data[label] = (tms, matrix)

    plot_groups(groups_data)


if __name__ == "__main__":
    main()
