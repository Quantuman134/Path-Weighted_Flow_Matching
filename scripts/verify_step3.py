#!/usr/bin/env python3
"""
Verify Step 3 (8-class convergence run) results.

Reads all validation outputs from experiment/tm_step3_8class_convergence/
and prints a clear pass/fail table with the gate criterion for proceeding
to Step 4 highlighted at the end.

Usage:
    python scripts/verify_step3.py
    python scripts/verify_step3.py --output experiment/tm_step3_8class_convergence
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict


GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def _ok(cond):
    return f"{GREEN}PASS{RESET}" if cond else f"{RED}FAIL{RESET}"


def _warn():
    return f"{YELLOW}WARN{RESET}"


def _read_csv_rows(path):
    with open(path) as f:
        return list(csv.reader(f))


def _print_section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def verify(base):
    print(f"Verifying Step 3 results in: {base}\n")
    all_pass = True

    # ------------------------------------------------------------------
    # 1. numerical_health.json
    # ------------------------------------------------------------------
    _print_section("1. Numerical health (numerical_health.json)")
    h = json.load(open(os.path.join(base, "numerical_health.json")))
    checks = [
        ("terminal w(T, T) ≈ 1.0",
         abs(h["terminal_w_avg_at_T_end"] - 1.0) < 1e-6,
         f"got {h['terminal_w_avg_at_T_end']:.10f}"),
        ("softmax row-sum dev < 1e-5",
         h["max_softmax_sum_deviation"] < 1e-5,
         f"got {h['max_softmax_sum_deviation']:.2e}"),
        ("no NaN or Inf",
         not h["any_nan_or_inf"],
         f"got any_nan_or_inf={h['any_nan_or_inf']}"),
    ]
    for name, ok, detail in checks:
        print(f"  [{_ok(ok)}]  {name:<40s} {detail}")
        all_pass &= ok

    # ------------------------------------------------------------------
    # 2. Autograd VJP
    # ------------------------------------------------------------------
    _print_section("2. Autograd VJP  (autograd_vjp_validation.csv)")
    print("   Test : analytic J^T a  vs  torch.autograd.grad(a·v, z)")
    print("   Bar  : rel err < 1e-8 (fp64)  or  < 1e-4 (fp32)")
    print()
    rows = _read_csv_rows(os.path.join(base, "autograd_vjp_validation.csv"))
    hdr = rows[0]
    i_cid   = hdr.index("class_id")
    i_t     = hdr.index("t")
    i_dtype = hdr.index("dtype")
    i_err   = hdr.index("rel_error_analytic_vs_autograd")

    n_pass = n_total = 0
    worst = 0.0
    for row in rows[1:]:
        err = float(row[i_err])
        dtype = row[i_dtype]
        threshold = 1e-8 if "float64" in dtype else 1e-4
        n_total += 1
        n_pass += (err < threshold)
        worst = max(worst, err)
    ok = n_pass == n_total
    print(f"   {n_pass}/{n_total} checks passed; worst rel_err = {worst:.3e}")
    print(f"   verdict: [{_ok(ok)}]")
    all_pass &= ok

    # ------------------------------------------------------------------
    # 3. FD trace validation
    # ------------------------------------------------------------------
    _print_section("3. FD trace agreement  (fd_trace_validation.csv)")
    print("   Test : forward-FD w  vs  backward-analytic w")
    print("   Bar  : best-δ rel_err < 0.05, plateau across δ (max/min < 10)")
    print()
    rows = _read_csv_rows(os.path.join(base, "fd_trace_validation.csv"))
    hdr = rows[0]
    delta_cols = [j for j, c in enumerate(hdr) if c.startswith("rel_err_delta_")]
    i_cid = hdr.index("class_id")

    n_pass = n_total = 0
    worst_min = 0.0
    for row in rows[1:]:
        deltas = [float(row[j]) for j in delta_cols]
        mn = min(deltas)
        mx = max(deltas)
        n_total += 1
        ok_row = (mn < 0.05) and ((mx / (mn + 1e-30)) < 10.0)
        n_pass += ok_row
        worst_min = max(worst_min, mn)
    ok = n_pass == n_total
    print(f"   {n_pass}/{n_total} classes passed; worst best-δ rel_err = "
          f"{worst_min:.3e}")
    print(f"   verdict: [{_ok(ok)}]")
    all_pass &= ok

    # ------------------------------------------------------------------
    # 4. Solver convergence  ←  the gate
    # ------------------------------------------------------------------
    _print_section("4. Solver convergence  (solver_convergence.csv)  ← GATE")
    print("   Test : w_avg(T, t) at each S vs S=1024 baseline")
    print("   GATE : at S=256, max_rel_err < 0.10  AND  mean_rel_err < 0.05")
    print("          (per doc §5.3; this is the critical Step 3 gate)")
    print()
    rows = _read_csv_rows(os.path.join(base, "solver_convergence.csv"))
    hdr = rows[0]
    i_cid  = hdr.index("class_id")
    i_S    = hdr.index("S")
    i_Sref = hdr.index("S_ref")
    i_max  = hdr.index("max_rel_err")
    i_mean = hdr.index("mean_rel_err")

    # Aggregate per S.
    by_S = defaultdict(list)
    for row in rows[1:]:
        S = int(row[i_S])
        by_S[S].append((float(row[i_max]), float(row[i_mean])))

    print(f"   {'S':>6}  {'max_rel_err (worst)':>22}  {'mean_rel_err (worst)':>22}"
          f"   verdict")
    gate_pass = None
    for S in sorted(by_S.keys()):
        vals = by_S[S]
        me_worst = max(v[0] for v in vals)
        mn_worst = max(v[1] for v in vals)
        if S == 256:
            ok = (me_worst < 0.10) and (mn_worst < 0.05)
            gate_pass = ok
            verdict = f"[{_ok(ok)}]  (GATE)"
            all_pass &= ok
        else:
            verdict = f"[{_warn()}]  (informational)"
        print(f"   {S:>6d}  {me_worst:>22.3e}  {mn_worst:>22.3e}   {verdict}")

    # ------------------------------------------------------------------
    # 5. Marginal moments
    # ------------------------------------------------------------------
    _print_section("5. Marginal moments  (marginal_moment_validation.csv)")
    print("   Test : integrated sample moments vs E[z_t]=t*y_bar, Cov(z_t)")
    print("   Bar  : mean_rel_err_MC < 3  AND  cov_trace_rel_err < 0.10")
    print()
    rows = _read_csv_rows(os.path.join(base, "marginal_moment_validation.csv"))
    hdr = rows[0]
    t_cols = [j for j, c in enumerate(hdr) if c.startswith("t=")]
    i_cid    = hdr.index("class_id")
    i_metric = hdr.index("metric")

    worst_mc = 0.0
    worst_cov = 0.0
    for row in rows[1:]:
        metric = row[i_metric]
        vals = [float(row[j]) for j in t_cols]
        if metric == "mean_rel_err_MC":
            worst_mc = max(worst_mc, max(vals))
        elif metric == "cov_trace_rel_err":
            worst_cov = max(worst_cov, max(vals))

    ok_mc  = worst_mc  < 3.0
    ok_cov = worst_cov < 0.10
    print(f"   worst mean_rel_err_MC   = {worst_mc:.3e}   [{_ok(ok_mc)}]  (< 3)")
    print(f"   worst cov_trace_rel_err = {worst_cov:.3e}   [{_ok(ok_cov)}]  (< 0.10)")
    all_pass &= ok_mc & ok_cov

    # ------------------------------------------------------------------
    # 6. Global-mean sanity (was catastrophic in v1)
    # ------------------------------------------------------------------
    _print_section("6. Global-mean magnitude  (wavg_global.csv)")
    print("   Test : global mean w_avg(t) should be O(1)–O(10^3),")
    print("          not O(10^4+) like the v1 run.")
    print("          Terminal value should be ≈ 1 at t = T_end.")
    print()
    with open(os.path.join(base, "wavg_global.csv")) as f:
        rows = list(csv.reader(f))
    hdr = rows[0]
    i_t = hdr.index("t")
    i_m = hdr.index("mean")
    means = [(float(r[i_t]), float(r[i_m])) for r in rows[1:]]
    max_mean = max(m for _, m in means)
    terminal_mean = means[-1][1]
    ok_mag = max_mean < 1e5
    ok_term = abs(terminal_mean - 1.0) < 1e-3
    print(f"   peak global mean = {max_mean:.3e}     [{_ok(ok_mag)}]  (< 1e5)")
    print(f"   terminal mean    = {terminal_mean:.6f}  [{_ok(ok_term)}]  (≈ 1)")
    print(f"   t = {means[0][0]:.4f} → {means[-1][0]:.4f}")
    print("   values along t (log scale, sampled):")
    step = max(1, len(means) // 8)
    for i in range(0, len(means), step):
        print(f"      t={means[i][0]:.4f}   mean={means[i][1]:.4e}")
    all_pass &= ok_mag & ok_term

    # ------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------
    _print_section("FINAL VERDICT")
    if all_pass:
        print(f"  {GREEN}All Step 3 acceptance criteria PASSED.{RESET}")
        print("  → Safe to proceed to Step 4 (32-class primary run).")
        return 0
    else:
        print(f"  {RED}One or more Step 3 checks FAILED (see above).{RESET}")
        if gate_pass is False:
            print(f"  {RED}The solver-convergence GATE at S=256 failed.{RESET}")
            print("  This is the critical criterion. Do NOT proceed to Step 4.")
        else:
            print("  → Investigate failing rows before Step 4.")
        return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="experiment/tm_step3_8class_convergence",
                    help="Output directory of the Step 3 8-class convergence run.")
    args = ap.parse_args()
    if not os.path.isdir(args.output):
        print(f"ERROR: directory not found: {args.output}", file=sys.stderr)
        sys.exit(2)
    sys.exit(verify(args.output))


if __name__ == "__main__":
    main()
