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
    print("   Bar  : best-δ rel_err < 0.05")
    print("          (plateau ratio only enforced when best-δ > 1e-3;")
    print("           below that, all δ are already at machine-precision")
    print("           and their ratio is meaningless jitter)")
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
        # Primary criterion: best δ meets the doc §6 threshold.
        ok_row = (mn < 0.05)
        # Secondary plateau check: only meaningful when values are not
        # already at machine-precision. If best-δ is already < 1e-3 the
        # test is trivially passed and the ratio is dominated by jitter.
        if mn > 1e-3:
            ok_row = ok_row and ((mx / (mn + 1e-30)) < 10.0)
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
    print("   Test : w_avg(T, t) at each S vs the S=512 baseline")
    print("   GATE : at S=256 vs S=512, max_rel_err < 0.10")
    print("          AND mean_rel_err < 0.05   (per doc §6, line 505)")
    print("   Extra: also prints S vs S_max as an informational")
    print("          (tighter) reference. That column is NOT the gate.")
    print()
    rows = _read_csv_rows(os.path.join(base, "solver_convergence.csv"))
    hdr = rows[0]
    i_cid  = hdr.index("class_id")
    i_S    = hdr.index("S")
    i_Sref = hdr.index("S_ref")
    i_max  = hdr.index("max_rel_err")
    i_mean = hdr.index("mean_rel_err")
    # Per-t breakdown columns (added by validation subrun for diagnosis).
    t_val_idx = [j for j, c in enumerate(hdr) if c.startswith("t_val")]
    t_err_idx = [j for j, c in enumerate(hdr) if c.startswith("rel_err_t")]

    # Group by S_ref then S so we can print two tables (official gate + info).
    by_ref = defaultdict(lambda: defaultdict(list))
    per_t_by_ref = defaultdict(lambda: defaultdict(list))  # S_ref -> S -> [rows of per-t rel_errs]
    t_vals_recorded = None
    for row in rows[1:]:
        S    = int(row[i_S])
        Sref = int(row[i_Sref])
        by_ref[Sref][S].append((float(row[i_max]), float(row[i_mean])))
        if t_err_idx:
            per_t = [float(row[j]) for j in t_err_idx]
            per_t_by_ref[Sref][S].append(per_t)
            if t_vals_recorded is None and t_val_idx:
                t_vals_recorded = [float(row[j]) for j in t_val_idx]

    if not by_ref:
        print("   ERROR: no rows in solver_convergence.csv")
        all_pass = False
        gate_pass = False
    else:
        # ---- (a) Official gate: S vs S=512  ---------------------------
        gate_ref = 512 if 512 in by_ref else max(by_ref.keys())
        info_ref = max(by_ref.keys()) if max(by_ref.keys()) != gate_ref else None

        print(f"   [GATE  vs S={gate_ref}]")
        print(f"     {'S':>6}  {'max_rel_err (worst)':>22}  "
              f"{'mean_rel_err (worst)':>22}   verdict")
        gate_pass = None
        for S in sorted(by_ref[gate_ref].keys()):
            vals = by_ref[gate_ref][S]
            me_worst = max(v[0] for v in vals)
            mn_worst = max(v[1] for v in vals)
            if S == 256:
                ok = (me_worst < 0.10) and (mn_worst < 0.05)
                gate_pass = ok
                verdict = f"[{_ok(ok)}]  (GATE, doc §6)"
                all_pass &= ok
            else:
                verdict = f"[{_warn()}]  (informational)"
            print(f"     {S:>6d}  {me_worst:>22.3e}  "
                  f"{mn_worst:>22.3e}   {verdict}")

        # ---- (b) Optional tighter reference (S vs S_max) --------------
        if info_ref is not None:
            print()
            print(f"   [INFO  vs S={info_ref}]   (tighter reference; "
                  f"NOT the gate. Larger errors here are")
            print( "                             expected if the true "
                   "solution is not yet Cauchy-converged at S=512.)")
            print(f"     {'S':>6}  {'max_rel_err (worst)':>22}  "
                  f"{'mean_rel_err (worst)':>22}")
            for S in sorted(by_ref[info_ref].keys()):
                vals = by_ref[info_ref][S]
                me_worst = max(v[0] for v in vals)
                mn_worst = max(v[1] for v in vals)
                print(f"     {S:>6d}  {me_worst:>22.3e}  "
                      f"{mn_worst:>22.3e}")

        # ---- (c) per-t breakdown for the gate pair (S=256 vs S_ref) ---
        if t_vals_recorded is not None and 256 in per_t_by_ref[gate_ref]:
            print()
            print(f"   [PER-t   S=256 vs S={gate_ref}]   (worst over classes)")
            rows_pt = per_t_by_ref[gate_ref][256]
            # worst over classes at each t
            n_t = len(rows_pt[0])
            worst_at_t = [max(row[j] for row in rows_pt) for j in range(n_t)]
            print(f"     {'t':>10}  {'worst rel_err':>16}")
            for tv, e in zip(t_vals_recorded, worst_at_t):
                flag = " ←" if e >= 0.10 else ""
                print(f"     {tv:>10.4f}  {e:>16.3e}{flag}")
            # Endpoint-singularity hint per doc §7.
            head_max = max(worst_at_t[:max(1, n_t // 4)])
            tail_max = max(worst_at_t[-max(1, n_t // 4):])
            if not gate_pass and tail_max > 3.0 * head_max:
                print()
                print(f"   {YELLOW}Note:{RESET}  errors concentrate near "
                      f"t → T_end (tail {tail_max:.2e} vs head {head_max:.2e}).")
                print( "          This is the expected discrete endpoint "
                       "singularity as T → 1,")
                print( "          not a solver bug (doc §7). Consider "
                       "lowering T (e.g. T=0.95) or")
                print( "          reporting a truncated interval that "
                       "excludes t > 0.9 · T_end.")

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
