#!/usr/bin/env python3
"""
Verify Step 2 (single-class fp64 diagnostic) results.

Reads all validation outputs from experiment/tm_step2_fp64_diagnostic/
and prints a clear pass/fail table.

Usage:
    python scripts/verify_step2.py
    python scripts/verify_step2.py --output experiment/tm_step2_fp64_diagnostic
"""

import argparse
import csv
import json
import os
import sys


# ANSI colors for the pass/fail column.
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
    print(f"Verifying Step 2 results in: {base}\n")

    all_pass = True

    # ------------------------------------------------------------------
    # 1. numerical_health.json
    # ------------------------------------------------------------------
    _print_section("1. Numerical health (numerical_health.json)")
    h_path = os.path.join(base, "numerical_health.json")
    with open(h_path) as f:
        h = json.load(f)

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
    # 2. Autograd VJP validation
    # ------------------------------------------------------------------
    _print_section("2. Autograd VJP  (autograd_vjp_validation.csv)")
    print("   Test: analytic J^T a  vs  torch.autograd.grad(a·v, z)")
    print("   Bar : rel err < 1e-8 (fp64)   or  < 1e-4 (fp32)")
    print()
    rows = _read_csv_rows(os.path.join(base, "autograd_vjp_validation.csv"))
    hdr = rows[0]
    i_t = hdr.index("t")
    i_dtype = hdr.index("dtype")
    i_err = hdr.index("rel_error_analytic_vs_autograd")

    print(f"   {'t':>8} {'dtype':>16} {'rel_error':>14}   verdict")
    for row in rows[1:]:
        t = float(row[i_t])
        dtype = row[i_dtype]
        err = float(row[i_err])
        threshold = 1e-8 if "float64" in dtype else 1e-4
        ok = err < threshold
        print(f"   {t:>8.4f} {dtype:>16} {err:>14.3e}   [{_ok(ok)}]  (< {threshold:.0e})")
        all_pass &= ok

    # ------------------------------------------------------------------
    # 3. Finite-difference trace validation
    # ------------------------------------------------------------------
    _print_section("3. FD trace agreement  (fd_trace_validation.csv)")
    print("   Test: backward-analytic  w = mean_k ||Phi^T q_k||^2")
    print("         vs forward-FD    w = mean_k ||(F(z+δq)-F(z-δq))/(2δ)||^2")
    print("   Bar : rel err < 0.05, and values should plateau across δ")
    print()
    rows = _read_csv_rows(os.path.join(base, "fd_trace_validation.csv"))
    hdr = rows[0]
    delta_cols = [(j, c) for j, c in enumerate(hdr)
                  if c.startswith("rel_err_delta_")]
    i_t = hdr.index("t")
    i_wb = hdr.index("w_back")

    print(f"   {'t':>8} {'w_back':>12}   " +
          " ".join(f"{c.replace('rel_err_delta_', 'δ='):>14}"
                   for _, c in delta_cols))
    for row in rows[1:]:
        t = float(row[i_t])
        wb = float(row[i_wb])
        deltas = [float(row[j]) for j, _ in delta_cols]
        min_err = min(deltas)
        max_err = max(deltas)
        # Best (lowest) δ column vs 0.05 tolerance
        ok_mag = min_err < 0.05
        # Plateau: max/min < 10x means the δ sweep is not diverging
        ok_plat = (max_err / (min_err + 1e-30)) < 10.0 if min_err > 0 else True
        ok = ok_mag and ok_plat
        print(f"   {t:>8.4f} {wb:>12.4e}   " +
              " ".join(f"{d:>14.3e}" for d in deltas) +
              f"   [{_ok(ok)}]")
        all_pass &= ok

    # ------------------------------------------------------------------
    # 4. Solver convergence
    # ------------------------------------------------------------------
    _print_section("4. Solver convergence  (solver_convergence.csv)")
    print("   Test: w_avg(T, t) computed at each S in {128, 256, 512}")
    print("         aligned in continuous time and compared to S=512")
    print("   Bar : at S=256, max_rel_err < 0.10  AND  mean_rel_err < 0.05")
    print()
    rows = _read_csv_rows(os.path.join(base, "solver_convergence.csv"))
    hdr = rows[0]
    i_S = hdr.index("S")
    i_Sref = hdr.index("S_ref")
    i_max = hdr.index("max_rel_err")
    i_mean = hdr.index("mean_rel_err")

    print(f"   {'S':>6} {'S_ref':>6} {'max_rel_err':>14} {'mean_rel_err':>14}   verdict")
    for row in rows[1:]:
        S = int(row[i_S])
        Sref = int(row[i_Sref])
        me = float(row[i_max])
        mn = float(row[i_mean])
        # Only S=256 vs S=512 is the gate; smaller S expected to be worse.
        if S == 256:
            ok = (me < 0.10) and (mn < 0.05)
            all_pass &= ok
            verdict = f"[{_ok(ok)}]  (this is the gate)"
        else:
            verdict = f"[{_warn()}]  (informational; S<256 expected worse)"
        print(f"   {S:>6d} {Sref:>6d} {me:>14.3e} {mn:>14.3e}   {verdict}")

    # ------------------------------------------------------------------
    # 5. Marginal moment validation
    # ------------------------------------------------------------------
    _print_section("5. Marginal moments  (marginal_moment_validation.csv)")
    print("   Test: sample moments of z_t match E[z_t]=t*y_bar,  Cov(z_t)")
    print("   Bar : mean_rel_err_MC < 3 at all t   (MC-normalized)")
    print("         cov_trace_rel_err < 0.10 at all t")
    print()
    rows = _read_csv_rows(os.path.join(base, "marginal_moment_validation.csv"))
    hdr = rows[0]
    t_cols = [(j, c) for j, c in enumerate(hdr) if c.startswith("t=")]
    t_vals = [float(c[2:]) for _, c in t_cols]
    i_metric = hdr.index("metric")

    for row in rows[1:]:
        metric = row[i_metric]
        vals = [float(row[j]) for j, _ in t_cols]
        vmax = max(vals)
        vmin = min(vals)

        if metric == "mean_rel_err_MC":
            ok = vmax < 3.0
            all_pass &= ok
            print(f"   {metric:<20s}   min={vmin:.3e}  max={vmax:.3e}   "
                  f"[{_ok(ok)}]  (bar: max < 3)")
        elif metric == "cov_trace_rel_err":
            ok = vmax < 0.10
            all_pass &= ok
            print(f"   {metric:<20s}   min={vmin:.3e}  max={vmax:.3e}   "
                  f"[{_ok(ok)}]  (bar: max < 0.10)")
        elif metric == "mean_rel_err_raw":
            print(f"   {metric:<20s}   min={vmin:.3e}  max={vmax:.3e}   "
                  f"[{_warn()}]  (informational; blows up at t≈0 by design)")
        else:
            print(f"   {metric:<20s}   min={vmin:.3e}  max={vmax:.3e}")

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    _print_section("FINAL VERDICT")
    if all_pass:
        print(f"  {GREEN}All Step 2 acceptance criteria PASSED.{RESET}")
        print("  → Safe to proceed to Step 3 (8-class convergence run).")
        return 0
    else:
        print(f"  {RED}One or more Step 2 checks FAILED (see above).{RESET}")
        print("  → Do NOT proceed to Step 3; report failing rows.")
        return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="experiment/tm_step2_fp64_diagnostic",
                    help="Output directory of the Step 2 diagnostic run.")
    args = ap.parse_args()
    if not os.path.isdir(args.output):
        print(f"ERROR: directory not found: {args.output}", file=sys.stderr)
        sys.exit(2)
    sys.exit(verify(args.output))


if __name__ == "__main__":
    main()
