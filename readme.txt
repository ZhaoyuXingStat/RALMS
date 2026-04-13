RALMS & CRALMS: Robust Adaptive Lasso with Multi-Signal Shrinkage
==================================================================

This repository provides standalone implementations of RALMS and CRALMS,
two estimators for recovering sparse binary signals from linear measurements
under heavy-tailed noise. These are companion code for the paper:

  [Robust Reconstruction of Latent Networks from Noisy Dynamics]
  [Zhaoyu Xing]
  [2026]


What are RALMS and CRALMS?
--------------------------

Both methods estimate a sparse vector x in {0,1}^p from the model

\begin{equation}  
    \bm{y}^t =  \left( \bm{A} \circ \bm{\Psi}^t \right) \bm{1}  +  \bm{\epsilon}^t
\end{equation}  

where Phi is an n x p design matrix and epsilon may follow a heavy-tailed
distribution (e.g., t or Cauchy).

- RALMS (Robust Adaptive Lasso with Multi-Signal Shrinkage):
  Unconstrained proximal gradient descent with quantile loss and an adaptive
  penalty

- CRALMS (Constrained RALMS):
  Adds a box constraint x_j in [0, 1] and solves via linearized ADMM.
  Typically produces cleaner binary estimates.

Both methods use adaptive weights computed from a quantile-Lasso initial
estimate. If no regularization parameter lambda is supplied, it is selected
automatically via BIC over a data-driven grid.


Repository Structure
--------------------

  R/
    RALMS.R                  RALMS function (R)
    CRALMS.R                 CRALMS function (R)
    test_RALMS_CRALMS.R      Test script (R)

  Python/
    RALMS.py                 RALMS function (Python, requires NumPy)
    CRALMS.py                CRALMS function (Python, requires NumPy)
    test_RALMS_CRALMS.py     Test script (Python)

  MATLAB/
    RALMS.m                  RALMS function (MATLAB)
    CRALMS.m                 CRALMS function (MATLAB)
    test_RALMS_CRALMS.m      Test script (MATLAB)


Quick Start
-----------

R:
    source("RALMS.R")
    source("CRALMS.R")
    result <- RALMS(y, Phi, tau = 0.5)
    result <- CRALMS(y, Phi, tau = 0.5)

Python:
    from RALMS import ralms
    from CRALMS import cralms
    result = ralms(y, Phi, tau=0.5)
    result = cralms(y, Phi, tau=0.5)

MATLAB:
    result = RALMS(y, Phi, 'tau', 0.5);
    result = CRALMS(y, Phi, 'tau', 0.5);


Parameters
----------

All three languages share the same interface:

  y          Observation vector (length n)
  Phi        Design matrix (n x p)
  tau        Quantile level in (0, 1), default 0.5
  lambda     Regularization parameter (positive). Auto-selected via BIC if
             not provided.
  gamma_w    Exponent for adaptive weights, default 1.0
  max_iter   Maximum iterations, default 1000
  tol        Convergence tolerance, default 1e-5
  rho        ADMM parameter (CRALMS only), default 1.0
  verbose    Print summary if true


Output
------

Each function returns a result containing:

  x_hat      Estimated signal vector (length p)
  weights    Adaptive weights used in penalization
  lambda     Lambda value used (selected or user-supplied)
  tau        Quantile level
  n_iter     Number of iterations
  converged  Whether the algorithm converged within max_iter
  rho        ADMM parameter (CRALMS only)


Running Tests
-------------

R:        Rscript test_RALMS_CRALMS.R
Python:   python test_RALMS_CRALMS.py
MATLAB:   run test_RALMS_CRALMS.m in the MATLAB command window


Citation
--------

If you use this code, please cite our paper. Thanks!
