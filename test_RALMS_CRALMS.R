# ==============================================================================
# Test Script for RALMS and CRALMS
# ==============================================================================
# This script verifies that both RALMS() and CRALMS() functions work correctly
# on synthetic data with known ground truth.
# ==============================================================================

cat("======================================================\n")
cat("  Test Suite for RALMS and CRALMS (R version)\n")
cat("======================================================\n\n")

source("RALMS.R")
source("CRALMS.R")

set.seed(42)
n_pass <- 0
n_fail <- 0

# Helper: report test result
report <- function(test_name, passed, msg = "") {
  if (passed) {
    cat(sprintf("  [PASS] %s\n", test_name))
    n_pass <<- n_pass + 1
  } else {
    cat(sprintf("  [FAIL] %s  -- %s\n", test_name, msg))
    n_fail <<- n_fail + 1
  }
}

# ==============================================================================
# Test 1: Basic functionality with simple synthetic data
# ==============================================================================
cat("--- Test 1: Basic estimation on synthetic data ---\n")

N <- 8; p <- choose(N, 2)   # p = 28 edges for 8 nodes
n <- N * 10                  # 80 observations

# True binary signal: ~30% edges present
x_true <- rep(0, p)
x_true[sample(p, round(0.3 * p))] <- 1

Phi <- matrix(rnorm(n * p), n, p)
y   <- as.vector(Phi %*% x_true) + rnorm(n, sd = 0.3)

# RALMS with auto-lambda
fit_r <- RALMS(y, Phi, tau = 0.5, verbose = TRUE)
report("RALMS returns list", is.list(fit_r))
report("RALMS x_hat length", length(fit_r$x_hat) == p)
report("RALMS lambda positive", fit_r$lambda > 0)

mse_r <- mean((x_true - fit_r$x_hat)^2)
cat(sprintf("  RALMS MSE = %.6f\n", mse_r))
report("RALMS converged or accurate", fit_r$converged || mse_r < 0.01)
report("RALMS MSE reasonable", mse_r < 0.5)

# CRALMS with auto-lambda
fit_c <- CRALMS(y, Phi, tau = 0.5, verbose = TRUE)
report("CRALMS returns list", is.list(fit_c))
report("CRALMS x_hat length", length(fit_c$x_hat) == p)
report("CRALMS x_hat in [0,1]", all(fit_c$x_hat >= -1e-10 & fit_c$x_hat <= 1 + 1e-10),
       sprintf("range [%.4f, %.4f]", min(fit_c$x_hat), max(fit_c$x_hat)))
report("CRALMS lambda positive", fit_c$lambda > 0)

mse_c <- mean((x_true - fit_c$x_hat)^2)
cat(sprintf("  CRALMS MSE = %.6f\n", mse_c))
report("CRALMS MSE reasonable", mse_c < 0.5)

# ==============================================================================
# Test 2: User-supplied lambda
# ==============================================================================
cat("\n--- Test 2: User-supplied lambda ---\n")

fit_r2 <- RALMS(y, Phi, tau = 0.5, lambda = 0.05)
report("RALMS user-lambda stored", abs(fit_r2$lambda - 0.05) < 1e-10)

fit_c2 <- CRALMS(y, Phi, tau = 0.5, lambda = 0.05)
report("CRALMS user-lambda stored", abs(fit_c2$lambda - 0.05) < 1e-10)

# ==============================================================================
# Test 3: Heavy-tailed noise (t-distribution, df=2)
# ==============================================================================
cat("\n--- Test 3: Heavy-tailed noise (t2) ---\n")

y_heavy <- as.vector(Phi %*% x_true) + rt(n, df = 2) * 0.5
fit_r3  <- RALMS(y_heavy, Phi, tau = 0.5, lambda = 0.1)
fit_c3  <- CRALMS(y_heavy, Phi, tau = 0.5, lambda = 0.1)

mse_r3 <- mean((x_true - fit_r3$x_hat)^2)
mse_c3 <- mean((x_true - fit_c3$x_hat)^2)
cat(sprintf("  RALMS  MSE (t2 noise) = %.6f\n", mse_r3))
cat(sprintf("  CRALMS MSE (t2 noise) = %.6f\n", mse_c3))
report("RALMS handles t2 noise", is.finite(mse_r3))
report("CRALMS handles t2 noise", is.finite(mse_c3))

# ==============================================================================
# Test 4: Different tau values
# ==============================================================================
cat("\n--- Test 4: Non-default tau ---\n")

fit_r4 <- RALMS(y, Phi, tau = 0.25, lambda = 0.1)
report("RALMS tau=0.25 works", fit_r4$tau == 0.25)

fit_c4 <- CRALMS(y, Phi, tau = 0.75, lambda = 0.1)
report("CRALMS tau=0.75 works", fit_c4$tau == 0.75)

# ==============================================================================
# Test 5: Input validation (should produce errors)
# ==============================================================================
cat("\n--- Test 5: Input validation ---\n")

# Dimension mismatch
err <- tryCatch(RALMS(y[1:5], Phi), error = function(e) e$message)
report("RALMS dim mismatch error", grepl("mismatch", err, ignore.case = TRUE), err)

# Invalid tau
err <- tryCatch(RALMS(y, Phi, tau = 1.5), error = function(e) e$message)
report("RALMS invalid tau error", grepl("tau", err, ignore.case = TRUE), err)

# Negative lambda
err <- tryCatch(CRALMS(y, Phi, lambda = -1), error = function(e) e$message)
report("CRALMS negative lambda error", grepl("lambda", err, ignore.case = TRUE), err)

# NA in y
y_na <- y; y_na[1] <- NA
err <- tryCatch(RALMS(y_na, Phi), error = function(e) e$message)
report("RALMS NA check", grepl("NA", err), err)

# ==============================================================================
# Test 6: Print methods
# ==============================================================================
cat("\n--- Test 6: Print methods ---\n")

cat("  -- RALMS print output --\n")
print(fit_r)

cat("  -- CRALMS print output --\n")
print(fit_c)

report("RALMS class correct", inherits(fit_r, "RALMS"))
report("CRALMS class correct", inherits(fit_c, "CRALMS"))

# ==============================================================================
# Test 7: Edge case - small problem
# ==============================================================================
cat("\n--- Test 7: Small problem (n=10, p=3) ---\n")

Phi_small <- matrix(rnorm(30), 10, 3)
x_small   <- c(1, 0, 1)
y_small   <- Phi_small %*% x_small + rnorm(10, sd = 0.1)

fit_rs <- RALMS(y_small, Phi_small, lambda = 0.01)
fit_cs <- CRALMS(y_small, Phi_small, lambda = 0.01)
report("RALMS small problem", length(fit_rs$x_hat) == 3)
report("CRALMS small problem", length(fit_cs$x_hat) == 3)
cat(sprintf("  x_true  = [%s]\n", paste(x_small, collapse = ", ")))
cat(sprintf("  RALMS   = [%s]\n", paste(sprintf("%.3f", fit_rs$x_hat), collapse = ", ")))
cat(sprintf("  CRALMS  = [%s]\n", paste(sprintf("%.3f", fit_cs$x_hat), collapse = ", ")))

# ==============================================================================
# Summary
# ==============================================================================
cat("\n======================================================\n")
cat(sprintf("  Results: %d passed, %d failed, %d total\n",
            n_pass, n_fail, n_pass + n_fail))
if (n_fail == 0) {
  cat("  ALL TESTS PASSED\n")
} else {
  cat("  SOME TESTS FAILED — please review above\n")
}
cat("======================================================\n")
