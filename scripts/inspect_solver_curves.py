#!/usr/bin/env python3
"""
Inspect raw w_avg convergence curves saved by the validation subrun.

The validation subrun writes wavg_convergence_classNNN.npz files with:
    target_times : (T_rep,)         common time grid
    S128, S256, S512, S1024, ...    per-S wavg curves on that grid

This script prints them side-by-side and computes absolute and relative
differences. Use it when scripts/verify_step3.py reports a solver-
convergence GATE failure, to see exactly where and how the curves differ.

Usage:
    python scripts/inspect_solver_curves.py \
        --output experiment/true_marginal_wavg_imagenet_T099_S256
"""

import argparse
import glob
import os
import sys

import numpy as np


def inspect(base):
    files = sorted(glob.glob(os.path.join(base,
                                          "wavg_convergence_class*.npz")))
    if not files:
        print(f"ERROR: no wavg_convergence_class*.npz in {base}",
              file=sys.stderr)
        return 1

    for path in files:
        cls_name = os.path.basename(path).replace(".npz", "")
        print("=" * 78)
        print(f"  {cls_name}")
        print("=" * 78)

        d = np.load(path)
        t = d["target_times"]
        S_keys = sorted(
            [k for k in d.files if k.startswith("S") and k != "S_ref"],
            key=lambda s: int(s[1:]),
        )

        # ---- absolute curves ----
        print()
        print("  Raw w_avg(T, t) curves at each S:")
        header = f"  {'t':>10}" + "".join(f"  {k:>14}" for k in S_keys)
        print(header)
        for i in range(len(t)):
            row = f"  {t[i]:>10.4f}"
            for k in S_keys:
                row += f"  {d[k][i]:>14.6e}"
            print(row)

        # ---- pairwise absolute and relative differences vs the largest S ----
        S_ref_key = S_keys[-1]                                # largest S
        ref = d[S_ref_key]
        print()
        print(f"  Absolute |curve - {S_ref_key}|:")
        print(f"  {'t':>10}" + "".join(f"  {k:>14}" for k in S_keys[:-1]))
        for i in range(len(t)):
            row = f"  {t[i]:>10.4f}"
            for k in S_keys[:-1]:
                row += f"  {abs(d[k][i] - ref[i]):>14.6e}"
            print(row)

        print()
        print(f"  Relative |curve - {S_ref_key}| / |{S_ref_key}|:")
        print(f"  {'t':>10}" + "".join(f"  {k:>14}" for k in S_keys[:-1]))
        for i in range(len(t)):
            row = f"  {t[i]:>10.4f}"
            for k in S_keys[:-1]:
                rel = abs(d[k][i] - ref[i]) / (abs(ref[i]) + 1e-30)
                row += f"  {rel:>14.6e}"
            print(row)

        # ---- pairwise vs S=512 (the doc §6 official reference) ----
        if "S512" in d.files:
            ref = d["S512"]
            print()
            print("  Relative |curve - S512| / |S512|  (doc §6 official gate):")
            print(f"  {'t':>10}" + "".join(f"  {k:>14}"
                                           for k in S_keys if k != "S512"))
            for i in range(len(t)):
                row = f"  {t[i]:>10.4f}"
                for k in S_keys:
                    if k == "S512":
                        continue
                    rel = abs(d[k][i] - ref[i]) / (abs(ref[i]) + 1e-30)
                    row += f"  {rel:>14.6e}"
                print(row)

        print()

    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",
                    default="experiment/true_marginal_wavg_imagenet_T099_S256",
                    help="Directory containing wavg_convergence_class*.npz.")
    args = ap.parse_args()
    if not os.path.isdir(args.output):
        print(f"ERROR: directory not found: {args.output}", file=sys.stderr)
        sys.exit(2)
    sys.exit(inspect(args.output))


if __name__ == "__main__":
    main()
