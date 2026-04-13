"""
==============================================================================
CRALMS: Constrained Robust Adaptive Lasso with Multi-Signal Shrinkage
==============================================================================

Estimates a sparse binary signal vector x in [0,1]^p from the linear model
    y = Phi @ x + epsilon
using quantile-loss with adaptive multi-signal penalization and a box
constraint x_j in [0, 1]. Solved via linearized ADMM.

Reference:
    [Your paper citation here]

Usage:
    from CRALMS import cralms
    result = cralms(y, Phi, tau=0.5)

==============================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class CRALMSResult:
    """Container for CRALMS estimation results."""
    x_hat: np.ndarray        # Estimated signal vector (length p, in [0,1])
    weights: np.ndarray      # Adaptive weights used
    lam: float               # Lambda value used
    tau: float               # Quantile level
    rho: float               # ADMM parameter
    n_iter: int              # Number of ADMM iterations
    converged: bool          # Whether algorithm converged

    def __repr__(self) -> str:
        p = len(self.x_hat)
        n_zero = int(np.sum(np.abs(self.x_hat) < 1e-4))
        n_one = int(np.sum(np.abs(self.x_hat - 1) < 1e-4))
        n_mid = p - n_zero - n_one

        lines = [
            "===== CRALMS Estimation Result =====",
            f"  tau (quantile level)   : {self.tau:.4f}",
            f"  lambda (regularization): {self.lam:.6f}",
            f"  rho (ADMM parameter)   : {self.rho:.4f}",
            f"  Iterations (ADMM)      : {self.n_iter}",
            f"  Converged              : {'Yes' if self.converged else 'No'}",
            f"  Signal dimension (p)   : {p}",
            f"  Estimated edges (|x~1|): {n_one}",
            f"  Estimated zeros (|x~0|): {n_zero}",
            f"  Intermediate values    : {n_mid}",
            f"  x_hat range            : [{self.x_hat.min():.4f}, {self.x_hat.max():.4f}]",
        ]
        if p <= 30:
            vals = "  ".join(f"{v:.4f}" for v in self.x_hat)
            lines.append(f"  x_hat: {vals}")
        else:
            vals = "  ".join(f"{v:.4f}" for v in self.x_hat[:20])
            lines.append(f"  x_hat (first 20): {vals}  ...")
        lines.append("====================================")
        return "\n".join(lines)


# ========================== Helper Functions ==================================

def _quantile_loss(u: np.ndarray, tau: float) -> float:
    return float(np.sum(u * (tau - (u < 0).astype(float))))


def _quantile_loss_subgradient(r: np.ndarray, tau: float) -> np.ndarray:
    return np.where(r > 0, tau, np.where(r < 0, tau - 1, tau - 0.5))


def _prox_multi_signal(v: float, lam_prime: float) -> float:
    """Proximal operator for penalty lambda * min(|x|, |x-1|)."""
    x1 = np.sign(v) * max(0.0, abs(v) - lam_prime)
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


def _quantile_lasso_initial(y, Phi, tau, lam, max_iter=500, tol=1e-4):
    """Standard quantile Lasso for initial estimate (adaptive weights)."""
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


def _calculate_bic(y, Phi, x_hat, tau):
    n = len(y)
    residuals = y - Phi @ x_hat
    loss_val = 2.0 * _quantile_loss(residuals, tau)
    k_active = int(np.sum((np.abs(x_hat) > 1e-4) & (np.abs(x_hat - 1) > 1e-4)))
    if k_active == 0:
        k_active = 0.5
    return n * np.log(max(loss_val / n, 1e-12)) + k_active * np.log(n)


def _compute_lambda_grid(y, Phi, tau, n_grid=20, ratio=1e-4):
    subgrad = _quantile_loss_subgradient(y, tau)
    lmax = float(np.max(np.abs(Phi.T @ subgrad)))
    if lmax == 0:
        lmax = 1.0
    return np.exp(np.linspace(np.log(lmax), np.log(lmax * ratio), n_grid))


# ========================== Main Function =====================================

def cralms(y: np.ndarray, Phi: np.ndarray,
           tau: float = 0.5,
           lam: Optional[float] = None,
           rho: float = 1.0,
           gamma_w: float = 1.0,
           max_iter: int = 1000,
           tol: float = 1e-5,
           verbose: bool = False) -> CRALMSResult:
    """
    CRALMS estimator (constrained, ADMM).

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
    rho : float
        ADMM augmented Lagrangian parameter. Default 1.0.
    gamma_w : float
        Exponent for adaptive weights. Default 1.0.
    max_iter : int
        Maximum ADMM iterations. Default 1000.
    tol : float
        Convergence tolerance. Default 1e-5.
    verbose : bool
        Print progress. Default False.

    Returns
    -------
    CRALMSResult
        Dataclass with x_hat, weights, lam, tau, rho, n_iter, converged.
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
    if rho <= 0:
        raise ValueError("'rho' must be positive.")
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

    # ====================== Core ADMM Solver ==================================
    N_INNER = 80  # Inner gradient steps per ADMM iteration

    def _solve(y, Phi, tau, lam_val, rho, gamma_w, max_iter, tol):
        # Adaptive weights
        lam_init = 0.1 * np.sqrt(np.log(max(p, 2)) / max(n, 1))
        x_init = _quantile_lasso_initial(y, Phi, tau, lam_init)
        weights = 1.0 / (np.minimum(np.abs(x_init), np.abs(x_init - 1)) + 0.01) ** gamma_w

        L_Phi = _estimate_L(Phi)
        x_hat = np.zeros(p)
        z_hat = np.zeros(p)
        u_hat = np.zeros(p)
        lr_inner = 1.0 / (L_Phi + rho)

        converged = False
        n_it = max_iter

        for k in range(1, max_iter + 1):
            z_old = z_hat.copy()
            v_x = z_hat - u_hat / rho

            # x-update: linearized proximal gradient
            for _ in range(N_INNER):
                residuals = y - Phi @ x_hat
                subgrad = _quantile_loss_subgradient(residuals, tau)
                grad_q = -(Phi.T @ subgrad)
                grad_r = rho * (x_hat - v_x)
                x_hat = x_hat - lr_inner * (grad_q + grad_r)

            # z-update: proximal + box projection [0, 1]
            v_z = x_hat + u_hat / rho
            for j in range(p):
                z_tilde = _prox_multi_signal(v_z[j], lam_val * weights[j] / rho)
                z_hat[j] = max(0.0, min(1.0, z_tilde))

            # dual update
            u_hat = u_hat + rho * (x_hat - z_hat)

            # convergence check
            primal_res = np.linalg.norm(x_hat - z_hat)
            dual_res = rho * np.linalg.norm(z_hat - z_old)
            if primal_res < tol and dual_res < tol:
                converged = True
                n_it = k
                break

        return z_hat, weights, n_it, converged

    # ====================== Lambda Selection ==================================
    if lam is None:
        if verbose:
            print("CRALMS: Selecting lambda via BIC ...")
        grid = _compute_lambda_grid(y, Phi, tau)
        bics = []
        for l in grid:
            xh, _, _, _ = _solve(y, Phi, tau, l, rho, gamma_w, max_iter, tol)
            bics.append(_calculate_bic(y, Phi, xh, tau))
        lam = float(grid[np.argmin(bics)])
        if verbose:
            print(f"  Selected lambda = {lam:.6f}")

    # ====================== Estimation ========================================
    x_hat, weights, n_it, conv = _solve(y, Phi, tau, lam, rho, gamma_w, max_iter, tol)

    result = CRALMSResult(
        x_hat=x_hat, weights=weights, lam=lam,
        tau=tau, rho=rho, n_iter=n_it, converged=conv
    )

    if verbose:
        print(result)

    return result
