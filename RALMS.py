"""
==============================================================================
RALMS: Robust Adaptive Lasso with Multi-Signal Shrinkage
==============================================================================

Estimates a sparse binary signal vector x in {0,1}^p from the linear model
    y = Phi @ x + epsilon
using quantile-loss with adaptive multi-signal penalization.
RALMS is the UNCONSTRAINED version (no box constraint on iterates).

Reference:
    [Your paper citation here]

Usage:
    from RALMS import ralms
    result = ralms(y, Phi, tau=0.5)

==============================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class RALMSResult:
    """Container for RALMS estimation results."""
    x_hat: np.ndarray        # Estimated signal vector (length p)
    weights: np.ndarray      # Adaptive weights used
    lam: float               # Lambda value used
    tau: float               # Quantile level
    n_iter: int              # Number of iterations
    converged: bool          # Whether algorithm converged

    def __repr__(self) -> str:
        p = len(self.x_hat)
        n_zero = int(np.sum(np.abs(self.x_hat) < 1e-4))
        n_one = int(np.sum(np.abs(self.x_hat - 1) < 1e-4))
        n_mid = p - n_zero - n_one

        lines = [
            "===== RALMS Estimation Result =====",
            f"  tau (quantile level)   : {self.tau:.4f}",
            f"  lambda (regularization): {self.lam:.6f}",
            f"  Iterations             : {self.n_iter}",
            f"  Converged              : {'Yes' if self.converged else 'No'}",
            f"  Signal dimension (p)   : {p}",
            f"  Estimated edges (|x~1|): {n_one}",
            f"  Estimated zeros (|x~0|): {n_zero}",
            f"  Intermediate values    : {n_mid}",
        ]
        if p <= 30:
            vals = "  ".join(f"{v:.4f}" for v in self.x_hat)
            lines.append(f"  x_hat: {vals}")
        else:
            vals = "  ".join(f"{v:.4f}" for v in self.x_hat[:20])
            lines.append(f"  x_hat (first 20): {vals}  ...")
        lines.append("===================================")
        return "\n".join(lines)


# ========================== Helper Functions ==================================

def _quantile_loss(u: np.ndarray, tau: float) -> float:
    """Check (quantile) loss function."""
    return float(np.sum(u * (tau - (u < 0).astype(float))))


def _quantile_loss_subgradient(r: np.ndarray, tau: float) -> np.ndarray:
    """Subgradient of the quantile loss."""
    grad = np.where(r > 0, tau, np.where(r < 0, tau - 1, tau - 0.5))
    return grad


def _prox_multi_signal(v: float, lam_prime: float) -> float:
    """Proximal operator for penalty lambda * min(|x|, |x-1|)."""
    # Candidate near 0
    x1 = np.sign(v) * max(0.0, abs(v) - lam_prime)
    # Candidate near 1
    x2 = np.sign(v - 1) * max(0.0, abs(v - 1) - lam_prime) + 1
    obj1 = 0.5 * (x1 - v) ** 2 + lam_prime * min(abs(x1), abs(x1 - 1))
    obj2 = 0.5 * (x2 - v) ** 2 + lam_prime * min(abs(x2), abs(x2 - 1))
    return x1 if obj1 <= obj2 else x2


def _estimate_L(Phi: np.ndarray, n_iter: int = 30) -> float:
    """Estimate largest eigenvalue of Phi'Phi via power iteration."""
    p = Phi.shape[1]
    v = np.random.randn(p)
    v /= np.linalg.norm(v)
    for _ in range(n_iter):
        Av = Phi.T @ (Phi @ v)
        nrm = np.linalg.norm(Av)
        if nrm < 1e-15:
            break
        v = Av / nrm
    return max(float(v @ (Phi.T @ (Phi @ v))), 1.0)


def _quantile_lasso_initial(y: np.ndarray, Phi: np.ndarray,
                            tau: float, lam: float,
                            max_iter: int = 500, tol: float = 1e-4) -> np.ndarray:
    """Standard quantile Lasso for initial estimate (used for adaptive weights)."""
    p = Phi.shape[1]
    x_hat = np.zeros(p)
    L = _estimate_L(Phi)
    lr = 1.0 / L
    for _ in range(max_iter):
        x_old = x_hat.copy()
        residuals = y - Phi @ x_hat
        grad = -Phi.T @ _quantile_loss_subgradient(residuals, tau)
        v = x_hat - lr * grad
        x_hat = np.sign(v) * np.maximum(0, np.abs(v) - lr * lam)
        if np.linalg.norm(x_hat - x_old) < tol:
            break
    return x_hat


def _calculate_bic(y: np.ndarray, Phi: np.ndarray,
                   x_hat: np.ndarray, tau: float) -> float:
    """BIC criterion for quantile regression."""
    n = len(y)
    residuals = y - Phi @ x_hat
    loss_val = 2.0 * _quantile_loss(residuals, tau)
    k_active = int(np.sum((np.abs(x_hat) > 1e-4) & (np.abs(x_hat - 1) > 1e-4)))
    if k_active == 0:
        k_active = 0.5
    return n * np.log(max(loss_val / n, 1e-12)) + k_active * np.log(n)


def _compute_lambda_grid(y: np.ndarray, Phi: np.ndarray, tau: float,
                         n_grid: int = 20, ratio: float = 1e-4) -> np.ndarray:
    """Data-driven lambda grid for quantile loss."""
    subgrad = _quantile_loss_subgradient(y, tau)
    lmax = float(np.max(np.abs(Phi.T @ subgrad)))
    if lmax == 0:
        lmax = 1.0
    return np.exp(np.linspace(np.log(lmax), np.log(lmax * ratio), n_grid))


# ========================== Main Function =====================================

def ralms(y: np.ndarray, Phi: np.ndarray,
          tau: float = 0.5,
          lam: Optional[float] = None,
          gamma_w: float = 1.0,
          max_iter: int = 1000,
          tol: float = 1e-5,
          verbose: bool = False) -> RALMSResult:
    """
    RALMS estimator (unconstrained).

    Parameters
    ----------
    y : ndarray, shape (n,)
        Response vector.
    Phi : ndarray, shape (n, p)
        Design / measurement matrix.
    tau : float
        Quantile level in (0, 1). Default 0.5.
    lam : float or None
        Regularization parameter. If None, selected via BIC.
    gamma_w : float
        Exponent for adaptive weights. Default 1.0.
    max_iter : int
        Maximum iterations. Default 1000.
    tol : float
        Convergence tolerance. Default 1e-5.
    verbose : bool
        Print progress. Default False.

    Returns
    -------
    RALMSResult
        Dataclass with x_hat, weights, lam, tau, n_iter, converged.
    """
    # ====================== Input Validation ==================================
    y = np.asarray(y, dtype=float).ravel()
    Phi = np.asarray(Phi, dtype=float)
    if Phi.ndim != 2:
        raise ValueError("'Phi' must be a 2D array (matrix).")

    n, p = Phi.shape
    if len(y) != n:
        raise ValueError(f"Dimension mismatch: len(y)={len(y)} but Phi has {n} rows.")
    if n < 2:
        raise ValueError("Need at least 2 observations (n >= 2).")
    if p < 1:
        raise ValueError("Need at least 1 predictor (p >= 1).")
    if not (0 < tau < 1):
        raise ValueError("'tau' must be in the open interval (0, 1).")
    if lam is not None and lam <= 0:
        raise ValueError("'lam' must be positive.")
    if gamma_w < 0:
        raise ValueError("'gamma_w' must be non-negative.")
    if max_iter < 1:
        raise ValueError("'max_iter' must be a positive integer.")
    if tol <= 0:
        raise ValueError("'tol' must be positive.")
    if np.any(np.isnan(y)) or np.any(np.isnan(Phi)):
        raise ValueError("Input contains NaN values.")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(Phi))):
        raise ValueError("Input contains non-finite values.")

    # ====================== Core Solver =======================================
    def _solve(y, Phi, tau, lam_val, gamma_w, max_iter, tol):
        # Adaptive weights
        lam_init = 0.1 * np.sqrt(np.log(max(p, 2)) / max(n, 1))
        x_init = _quantile_lasso_initial(y, Phi, tau, lam_init)
        weights = 1.0 / (np.minimum(np.abs(x_init), np.abs(x_init - 1)) + 0.01) ** gamma_w

        L_Phi = _estimate_L(Phi)
        x_hat = np.zeros(p)
        lr = 1.0 / L_Phi
        converged = False
        n_it = max_iter

        for k in range(1, max_iter + 1):
            x_old = x_hat.copy()
            residuals = y - Phi @ x_hat
            subgrad = _quantile_loss_subgradient(residuals, tau)
            grad = -(Phi.T @ subgrad)
            v = x_hat - lr * grad
            for j in range(p):
                x_hat[j] = _prox_multi_signal(v[j], lr * lam_val * weights[j])
            if np.linalg.norm(x_hat - x_old) < tol:
                converged = True
                n_it = k
                break

        return x_hat, weights, n_it, converged

    # ====================== Lambda Selection ==================================
    if lam is None:
        if verbose:
            print("RALMS: Selecting lambda via BIC ...")
        grid = _compute_lambda_grid(y, Phi, tau)
        bics = []
        for l in grid:
            xh, _, _, _ = _solve(y, Phi, tau, l, gamma_w, max_iter, tol)
            bics.append(_calculate_bic(y, Phi, xh, tau))
        lam = float(grid[np.argmin(bics)])
        if verbose:
            print(f"  Selected lambda = {lam:.6f}")

    # ====================== Estimation ========================================
    x_hat, weights, n_it, conv = _solve(y, Phi, tau, lam, gamma_w, max_iter, tol)

    result = RALMSResult(
        x_hat=x_hat, weights=weights, lam=lam,
        tau=tau, n_iter=n_it, converged=conv
    )

    if verbose:
        print(result)

    return result
