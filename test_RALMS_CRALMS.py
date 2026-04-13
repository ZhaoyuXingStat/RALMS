"""
==============================================================================
Test Script for RALMS and CRALMS (Python version)
==============================================================================
Verifies that both ralms() and cralms() functions work correctly on
synthetic data with known ground truth.
==============================================================================
"""

import numpy as np
import sys

from RALMS import ralms, RALMSResult
from CRALMS import cralms, CRALMSResult

np.random.seed(42)

n_pass = 0
n_fail = 0


def report(test_name: str, passed: bool, msg: str = ""):
    global n_pass, n_fail
    if passed:
        print(f"  [PASS] {test_name}")
        n_pass += 1
    else:
        print(f"  [FAIL] {test_name}  -- {msg}")
        n_fail += 1


print("=" * 56)
print("  Test Suite for RALMS and CRALMS (Python version)")
print("=" * 56)
print()

# ==============================================================================
# Test 1: Basic functionality with simple synthetic data
# ==============================================================================
print("--- Test 1: Basic estimation on synthetic data ---")

N = 8
p = N * (N - 1) // 2  # 28 edges
n = N * 10             # 80 observations

x_true = np.zeros(p)
idx = np.random.choice(p, size=int(0.3 * p), replace=False)
x_true[idx] = 1.0

Phi = np.random.randn(n, p)
y = Phi @ x_true + np.random.randn(n) * 0.3

# RALMS with auto-lambda
fit_r = ralms(y, Phi, tau=0.5, verbose=True)
report("RALMS returns RALMSResult", isinstance(fit_r, RALMSResult))
report("RALMS x_hat length", len(fit_r.x_hat) == p)
report("RALMS lambda positive", fit_r.lam > 0)

mse_r = float(np.mean((x_true - fit_r.x_hat) ** 2))
print(f"  RALMS MSE = {mse_r:.6f}")
report("RALMS converged or accurate", fit_r.converged or mse_r < 0.01)
report("RALMS MSE reasonable", mse_r < 0.5)

# CRALMS with auto-lambda
fit_c = cralms(y, Phi, tau=0.5, verbose=True)
report("CRALMS returns CRALMSResult", isinstance(fit_c, CRALMSResult))
report("CRALMS x_hat length", len(fit_c.x_hat) == p)
report("CRALMS x_hat in [0,1]",
       bool(np.all(fit_c.x_hat >= -1e-10) and np.all(fit_c.x_hat <= 1 + 1e-10)),
       f"range [{fit_c.x_hat.min():.4f}, {fit_c.x_hat.max():.4f}]")
report("CRALMS lambda positive", fit_c.lam > 0)

mse_c = float(np.mean((x_true - fit_c.x_hat) ** 2))
print(f"  CRALMS MSE = {mse_c:.6f}")
report("CRALMS MSE reasonable", mse_c < 0.5)

# ==============================================================================
# Test 2: User-supplied lambda
# ==============================================================================
print("\n--- Test 2: User-supplied lambda ---")

fit_r2 = ralms(y, Phi, tau=0.5, lam=0.05)
report("RALMS user-lambda stored", abs(fit_r2.lam - 0.05) < 1e-10)

fit_c2 = cralms(y, Phi, tau=0.5, lam=0.05)
report("CRALMS user-lambda stored", abs(fit_c2.lam - 0.05) < 1e-10)

# ==============================================================================
# Test 3: Heavy-tailed noise (t-distribution, df=2)
# ==============================================================================
print("\n--- Test 3: Heavy-tailed noise (t2) ---")

# np.random uses standard_t for t-distribution
y_heavy = Phi @ x_true + np.random.standard_t(df=2, size=n) * 0.5
fit_r3 = ralms(y_heavy, Phi, tau=0.5, lam=0.1)
fit_c3 = cralms(y_heavy, Phi, tau=0.5, lam=0.1)

mse_r3 = float(np.mean((x_true - fit_r3.x_hat) ** 2))
mse_c3 = float(np.mean((x_true - fit_c3.x_hat) ** 2))
print(f"  RALMS  MSE (t2 noise) = {mse_r3:.6f}")
print(f"  CRALMS MSE (t2 noise) = {mse_c3:.6f}")
report("RALMS handles t2 noise", np.isfinite(mse_r3))
report("CRALMS handles t2 noise", np.isfinite(mse_c3))

# ==============================================================================
# Test 4: Different tau values
# ==============================================================================
print("\n--- Test 4: Non-default tau ---")

fit_r4 = ralms(y, Phi, tau=0.25, lam=0.1)
report("RALMS tau=0.25 works", fit_r4.tau == 0.25)

fit_c4 = cralms(y, Phi, tau=0.75, lam=0.1)
report("CRALMS tau=0.75 works", fit_c4.tau == 0.75)

# ==============================================================================
# Test 5: Input validation (should produce errors)
# ==============================================================================
print("\n--- Test 5: Input validation ---")

# Dimension mismatch
try:
    ralms(y[:5], Phi)
    report("RALMS dim mismatch error", False, "No error raised")
except ValueError as e:
    report("RALMS dim mismatch error", "mismatch" in str(e).lower())

# Invalid tau
try:
    ralms(y, Phi, tau=1.5)
    report("RALMS invalid tau error", False, "No error raised")
except ValueError as e:
    report("RALMS invalid tau error", "tau" in str(e).lower())

# Negative lambda
try:
    cralms(y, Phi, lam=-1)
    report("CRALMS negative lambda error", False, "No error raised")
except ValueError as e:
    report("CRALMS negative lambda error", "lam" in str(e).lower())

# NaN in y
y_nan = y.copy()
y_nan[0] = np.nan
try:
    ralms(y_nan, Phi)
    report("RALMS NaN check", False, "No error raised")
except ValueError as e:
    report("RALMS NaN check", "nan" in str(e).lower())

# ==============================================================================
# Test 6: Print / repr methods
# ==============================================================================
print("\n--- Test 6: Print / repr methods ---")
print("  -- RALMS repr --")
print(fit_r)
print("  -- CRALMS repr --")
print(fit_c)
report("RALMS repr works", "RALMS" in repr(fit_r))
report("CRALMS repr works", "CRALMS" in repr(fit_c))

# ==============================================================================
# Test 7: Small problem
# ==============================================================================
print("\n--- Test 7: Small problem (n=10, p=3) ---")

Phi_small = np.random.randn(10, 3)
x_small = np.array([1.0, 0.0, 1.0])
y_small = Phi_small @ x_small + np.random.randn(10) * 0.1

fit_rs = ralms(y_small, Phi_small, lam=0.01)
fit_cs = cralms(y_small, Phi_small, lam=0.01)
report("RALMS small problem", len(fit_rs.x_hat) == 3)
report("CRALMS small problem", len(fit_cs.x_hat) == 3)
print(f"  x_true  = {x_small}")
print(f"  RALMS   = {np.round(fit_rs.x_hat, 3)}")
print(f"  CRALMS  = {np.round(fit_cs.x_hat, 3)}")

# ==============================================================================
# Summary
# ==============================================================================
print()
print("=" * 56)
print(f"  Results: {n_pass} passed, {n_fail} failed, {n_pass + n_fail} total")
if n_fail == 0:
    print("  ALL TESTS PASSED")
else:
    print("  SOME TESTS FAILED — please review above")
print("=" * 56)

sys.exit(0 if n_fail == 0 else 1)
