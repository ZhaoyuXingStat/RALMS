# ==============================================================================
# RALMS: Robust Adaptive Lasso with Multi-Signal Shrinkage
# ==============================================================================
#
# Description:
#   Estimates a sparse binary signal vector x in {0,1}^p from the linear model
#       y = Phi * x + epsilon
#   using quantile-loss with adaptive multi-signal penalization.
#   RALMS is the UNCONSTRAINED version (no box constraint on iterates).
#
# Reference:
#   [Your paper citation here]
#
# Usage:
#   result <- RALMS(y, Phi, tau = 0.5, lambda = NULL, gamma_w = 1.0,
#                   max_iter = 1000, tol = 1e-5, verbose = FALSE)
#
# Arguments:
#   y        : numeric vector of length n (response / observations)
#   Phi      : numeric matrix of dimension n x p (design / measurement matrix)
#   tau      : quantile level in (0, 1), default 0.5 (median regression)
#   lambda   : regularization parameter (positive scalar). If NULL, selected
#              automatically via BIC over a data-driven grid.
#   gamma_w  : exponent for adaptive weights, default 1.0
#   max_iter : maximum number of proximal-gradient iterations, default 1000
#   tol      : convergence tolerance, default 1e-5
#   verbose  : if TRUE, print iteration progress
#
# Returns:
#   A list with components:
#     x_hat    : numeric vector of length p, the estimated signal
#     weights  : adaptive weights used in penalization
#     lambda   : the lambda value used (selected or user-supplied)
#     tau      : quantile level used
#     n_iter   : number of iterations until convergence
#     converged: logical, whether the algorithm converged within max_iter
#
# ==============================================================================

RALMS <- function(y, Phi, tau = 0.5, lambda = NULL, gamma_w = 1.0,
                  max_iter = 1000, tol = 1e-5, verbose = FALSE) {

  # ========================== Input Validation ================================
  if (!is.numeric(y)) stop("'y' must be a numeric vector.")
  if (!is.numeric(Phi) && !is.matrix(Phi)) stop("'Phi' must be a numeric matrix.")
  Phi <- as.matrix(Phi)
  y   <- as.numeric(y)

  n <- nrow(Phi); p <- ncol(Phi)
  if (length(y) != n) {
    stop(sprintf("Dimension mismatch: length(y)=%d but nrow(Phi)=%d.", length(y), n))
  }
  if (n < 2) stop("Need at least 2 observations (n >= 2).")
  if (p < 1) stop("Need at least 1 predictor (p >= 1).")

  if (tau <= 0 || tau >= 1) stop("'tau' must be in the open interval (0, 1).")
  if (!is.null(lambda) && (lambda <= 0)) stop("'lambda' must be positive.")
  if (gamma_w < 0) stop("'gamma_w' must be non-negative.")
  if (max_iter < 1) stop("'max_iter' must be a positive integer.")
  if (tol <= 0) stop("'tol' must be positive.")

  if (any(is.na(y))) stop("'y' contains NA values.")
  if (any(is.na(Phi))) stop("'Phi' contains NA values.")
  if (any(!is.finite(y))) stop("'y' contains non-finite values.")
  if (any(!is.finite(Phi))) stop("'Phi' contains non-finite values.")

  # ========================== Helper Functions ================================

  quantile_loss <- function(u, tau) sum(u * (tau - (u < 0)))

  quantile_loss_subgradient <- function(r, tau) {
    grad <- numeric(length(r))
    grad[r > 0]  <- tau
    grad[r < 0]  <- tau - 1
    grad[r == 0] <- tau - 0.5
    grad
  }

  prox_multi_signal <- function(v, lambda_prime) {
    # Proximal operator for the multi-signal penalty: lambda * min(|x|, |x-1|)
    x1 <- sign(v) * max(0, abs(v) - lambda_prime)
    x2 <- sign(v - 1) * max(0, abs(v - 1) - lambda_prime) + 1
    obj1 <- 0.5 * (x1 - v)^2 + lambda_prime * min(abs(x1), abs(x1 - 1))
    obj2 <- 0.5 * (x2 - v)^2 + lambda_prime * min(abs(x2), abs(x2 - 1))
    if (obj1 <= obj2) x1 else x2
  }

  estimate_L <- function(Phi, n_iter = 30) {
    # Power iteration to estimate the largest eigenvalue of Phi'Phi
    pp <- ncol(Phi); v <- rnorm(pp); v <- v / sqrt(sum(v^2))
    f <- function(x) as.vector(crossprod(Phi, Phi %*% x))
    for (i in 1:n_iter) {
      Av <- f(v); nrm <- sqrt(sum(Av^2))
      if (nrm < 1e-15) break
      v <- Av / nrm
    }
    max(sum(v * f(v)), 1)
  }

  quantile_lasso_initial <- function(y, Phi, tau, lam, max_it = 500, tol_init = 1e-4) {
    # Standard quantile Lasso for computing initial estimate (used for weights)
    pp <- ncol(Phi); x_hat <- rep(0, pp); L <- estimate_L(Phi); lr <- 1 / L
    for (k in 1:max_it) {
      x_old <- x_hat
      residuals <- y - Phi %*% x_hat
      grad <- -crossprod(Phi, quantile_loss_subgradient(residuals, tau))
      v <- x_hat - lr * as.vector(grad)
      x_hat <- sign(v) * pmax(0, abs(v) - lr * lam)
      if (sqrt(sum((x_hat - x_old)^2)) < tol_init) break
    }
    x_hat
  }

  calculate_bic <- function(y, Phi, x_hat, tau) {
    nn <- length(y)
    residuals <- as.vector(y - Phi %*% x_hat)
    loss_val <- 2 * quantile_loss(residuals, tau)
    k_active <- sum(abs(x_hat) > 1e-4 & abs(x_hat - 1) > 1e-4)
    if (k_active == 0) k_active <- 0.5
    nn * log(max(loss_val / nn, 1e-12)) + k_active * log(nn)
  }

  compute_lambda_grid <- function(y, Phi, tau, n_grid = 20, ratio = 1e-4) {
    subgrad <- quantile_loss_subgradient(y, tau)
    lmax <- max(abs(as.vector(crossprod(Phi, subgrad))))
    if (lmax == 0) lmax <- 1
    exp(seq(log(lmax), log(lmax * ratio), length.out = n_grid))
  }

  # ========================== Core RALMS Solver ===============================

  ralms_solve <- function(y, Phi, tau, lambda, gamma_w, max_iter, tol) {
    pp <- ncol(Phi)

    # Adaptive weights from quantile Lasso initial estimator
    lambda_init <- 0.1 * sqrt(log(max(pp, 2)) / max(n, 1))
    x_init <- quantile_lasso_initial(y, Phi, tau, lam = lambda_init)
    weights <- 1 / (pmin(abs(x_init), abs(x_init - 1)) + 0.01)^gamma_w

    # Proximal gradient descent (unconstrained)
    L_Phi <- estimate_L(Phi)
    x_hat <- rep(0, pp)
    lr <- 1 / L_Phi
    converged <- FALSE
    n_iter <- max_iter

    for (k in 1:max_iter) {
      x_old <- x_hat
      residuals <- as.vector(y - Phi %*% x_hat)
      subgrad <- quantile_loss_subgradient(residuals, tau)
      grad <- -as.vector(crossprod(Phi, subgrad))
      v <- x_hat - lr * grad
      for (j in 1:pp) {
        x_hat[j] <- prox_multi_signal(v[j], lr * lambda * weights[j])
      }
      if (sqrt(sum((x_hat - x_old)^2)) < tol) {
        converged <- TRUE; n_iter <- k; break
      }
    }

    list(x_hat = x_hat, weights = weights, n_iter = n_iter, converged = converged)
  }

  # ========================== Lambda Selection (if needed) ====================

  if (is.null(lambda)) {
    if (verbose) cat("RALMS: Selecting lambda via BIC ...\n")
    grid <- compute_lambda_grid(y, Phi, tau, n_grid = 20, ratio = 1e-4)
    bics <- sapply(grid, function(lam) {
      fit <- ralms_solve(y, Phi, tau, lam, gamma_w, max_iter, tol)
      calculate_bic(y, Phi, fit$x_hat, tau)
    })
    lambda <- grid[which.min(bics)]
    if (verbose) cat(sprintf("  Selected lambda = %.6f\n", lambda))
  }

  # ========================== Main Estimation =================================

  result <- ralms_solve(y, Phi, tau, lambda, gamma_w, max_iter, tol)

  out <- list(
    x_hat     = result$x_hat,
    weights   = result$weights,
    lambda    = lambda,
    tau       = tau,
    n_iter    = result$n_iter,
    converged = result$converged
  )
  class(out) <- "RALMS"

  # ========================== Print Summary ===================================

  if (verbose) {
    cat("\n")
    print(out)
  }

  return(out)
}


# ==============================================================================
# Print method for RALMS objects
# ==============================================================================
print.RALMS <- function(x, ...) {
  cat("===== RALMS Estimation Result =====\n")
  cat(sprintf("  tau (quantile level)   : %.4f\n", x$tau))
  cat(sprintf("  lambda (regularization): %.6f\n", x$lambda))
  cat(sprintf("  Iterations             : %d\n", x$n_iter))
  cat(sprintf("  Converged              : %s\n", ifelse(x$converged, "Yes", "No")))

  p <- length(x$x_hat)
  n_zero <- sum(abs(x$x_hat) < 1e-4)
  n_one  <- sum(abs(x$x_hat - 1) < 1e-4)
  n_mid  <- p - n_zero - n_one
  cat(sprintf("  Signal dimension (p)   : %d\n", p))
  cat(sprintf("  Estimated edges (|x~1|): %d\n", n_one))
  cat(sprintf("  Estimated zeros (|x~0|): %d\n", n_zero))
  cat(sprintf("  Intermediate values    : %d\n", n_mid))

  if (p <= 30) {
    cat("  x_hat: ")
    cat(sprintf("%.4f", x$x_hat), sep = "  ")
    cat("\n")
  } else {
    cat("  x_hat (first 20): ")
    cat(sprintf("%.4f", x$x_hat[1:20]), sep = "  ")
    cat("  ...\n")
  }
  cat("===================================\n")
  invisible(x)
}
