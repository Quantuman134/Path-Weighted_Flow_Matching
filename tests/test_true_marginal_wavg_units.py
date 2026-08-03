"""
Unit tests for exp_true_marginal_wavg.py -- synthetic-dimension checks
(revision doc, section 5, step 1).

Verifies:
  1. Responsibility rows sum to 1 (< 1e-5 deviation).
  2. Analytic  J_v^T a  matches autograd  d/dz (a . v(z, t))  to < 1e-4 in
     FP32 and < 1e-8 in FP64.
  3. Batched  C_alpha a  matches an explicit (D x D) reference to < 1e-8
     in FP64.

Run with:
    pytest -q tests/test_true_marginal_wavg_units.py
Or standalone:
    python tests/test_true_marginal_wavg_units.py
"""

import os
import sys

import numpy as np
import pytest
import torch

# Make the top-level module importable when run either from repo root or
# from within tests/.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import exp_true_marginal_wavg as tm    # noqa: E402


def _device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# 1. Responsibility normalization
# ------------------------------------------------------------------
@pytest.mark.parametrize("D,M,N", [(16, 32, 8), (64, 128, 16)])
@pytest.mark.parametrize("t", [0.10, 0.50, 0.90, 0.99])
def test_responsibility_row_sum(D, M, N, t):
    dev = _device()
    torch.manual_seed(0)
    Y = torch.randn(M, D, device=dev, dtype=torch.float64)
    Z = torch.randn(N, D, device=dev, dtype=torch.float64)
    y_sq = (Y * Y).sum(dim=1, keepdim=True).T
    alpha, mu = tm.responsibilities(Z, Y, y_sq, t)
    row_sums = alpha.sum(dim=1)
    dev_max = float((row_sums - 1.0).abs().max().item())
    assert dev_max < 1e-5, f"row-sum deviation {dev_max}"


# ------------------------------------------------------------------
# 2. Analytic J^T a vs autograd
# ------------------------------------------------------------------
def _analytic_vs_autograd(D, M, K_val, N_val, t, dtype):
    dev = _device()
    torch.manual_seed(1)
    Y = torch.randn(M, D, device=dev, dtype=dtype)
    Z = torch.randn(N_val, D, device=dev, dtype=dtype)
    A = torch.randn(K_val, N_val, D, device=dev, dtype=dtype)
    y_sq = (Y * Y).sum(dim=1, keepdim=True).T

    with torch.no_grad():
        alpha, mu = tm.responsibilities(Z, Y, y_sq, t)
        JT_analytic = tm.apply_JT_batched(A, Y, t, alpha, mu)

    JT_autograd = torch.zeros_like(JT_analytic)
    for n in range(N_val):
        for k in range(K_val):
            z = Z[n].detach().clone().requires_grad_(True)
            v = tm._velocity_diff(z, Y, t)
            scalar = (A[k, n] * v).sum()
            g = torch.autograd.grad(scalar, z)[0]
            JT_autograd[k, n] = g

    diff = (JT_analytic - JT_autograd).to(torch.float64)
    ref = JT_autograd.to(torch.float64)
    return float((diff.norm() / (ref.norm() + 1e-30)).item())


@pytest.mark.parametrize("t", [0.25, 0.50, 0.75])
def test_analytic_vs_autograd_fp32(t):
    err = _analytic_vs_autograd(D=64, M=32, K_val=4, N_val=4, t=t,
                                dtype=torch.float32)
    assert err < 1e-4, f"FP32 rel err {err} at t={t}"


@pytest.mark.parametrize("t", [0.25, 0.50, 0.75])
def test_analytic_vs_autograd_fp64(t):
    err = _analytic_vs_autograd(D=64, M=32, K_val=4, N_val=4, t=t,
                                dtype=torch.float64)
    assert err < 1e-8, f"FP64 rel err {err} at t={t}"


# ------------------------------------------------------------------
# 3. Batched vs explicit C_alpha
# ------------------------------------------------------------------
@pytest.mark.parametrize("t", [0.25, 0.50, 0.75, 0.95])
def test_batched_vs_explicit_cov_action(t):
    dev = _device()
    torch.manual_seed(2)
    D, M = 32, 24
    Y = torch.randn(M, D, device=dev, dtype=torch.float64)
    res = tm.cov_action_reference_fp64(Y, t=t, seed=42)
    assert res["rel_error_batched_vs_explicit"] < 1e-8, res


if __name__ == "__main__":
    # Simple standalone runner (skip pytest CLI).
    for D, M, N in [(16, 32, 8), (64, 128, 16)]:
        for t in [0.10, 0.50, 0.90, 0.99]:
            test_responsibility_row_sum(D, M, N, t)
    for t in [0.25, 0.50, 0.75]:
        test_analytic_vs_autograd_fp32(t)
        test_analytic_vs_autograd_fp64(t)
    for t in [0.25, 0.50, 0.75, 0.95]:
        test_batched_vs_explicit_cov_action(t)
    print("All unit tests passed.")
