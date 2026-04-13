function result = RALMS(y, Phi, varargin)
% RALMS  Robust Adaptive Lasso with Multi-Signal Shrinkage (unconstrained)
%
%   result = RALMS(y, Phi)
%   result = RALMS(y, Phi, 'tau', 0.5, 'lambda', 0.1, ...)
%
%   Estimates a sparse binary signal vector x in {0,1}^p from the model
%       y = Phi * x + epsilon
%   using quantile-loss with adaptive multi-signal penalization.
%
%   Inputs:
%     y   : column vector of length n (observations)
%     Phi : matrix of size n x p (design / measurement matrix)
%
%   Name-Value Parameters:
%     'tau'      - quantile level in (0,1), default 0.5
%     'lambda'   - regularization parameter (positive). If empty/NaN,
%                  selected automatically via BIC. Default: [] (auto)
%     'gamma_w'  - exponent for adaptive weights, default 1.0
%     'max_iter' - maximum iterations, default 1000
%     'tol'      - convergence tolerance, default 1e-5
%     'verbose'  - logical, print progress, default false
%
%   Output:
%     result : struct with fields
%       .x_hat     - estimated signal (p x 1)
%       .weights   - adaptive weights (p x 1)
%       .lambda    - lambda used
%       .tau       - quantile level used
%       .n_iter    - iterations used
%       .converged - logical
%
%   Reference:
%     [Your paper citation here]

    % ======================== Parse Inputs ====================================
    p = inputParser;
    addRequired(p, 'y', @isnumeric);
    addRequired(p, 'Phi', @isnumeric);
    addParameter(p, 'tau', 0.5, @isnumeric);
    addParameter(p, 'lambda', [], @isnumeric);
    addParameter(p, 'gamma_w', 1.0, @isnumeric);
    addParameter(p, 'max_iter', 1000, @isnumeric);
    addParameter(p, 'tol', 1e-5, @isnumeric);
    addParameter(p, 'verbose', false, @islogical);
    parse(p, y, Phi, varargin{:});
    opts = p.Results;

    % ======================== Input Validation ================================
    y = y(:);  % force column
    [n, p_dim] = size(Phi);

    if length(y) ~= n
        error('RALMS:dimMismatch', ...
              'Dimension mismatch: length(y)=%d but size(Phi,1)=%d.', length(y), n);
    end
    if n < 2, error('RALMS:tooFewObs', 'Need at least 2 observations.'); end
    if p_dim < 1, error('RALMS:noPredictors', 'Need at least 1 predictor.'); end
    if opts.tau <= 0 || opts.tau >= 1
        error('RALMS:invalidTau', 'tau must be in (0, 1).');
    end
    if ~isempty(opts.lambda) && opts.lambda <= 0
        error('RALMS:invalidLambda', 'lambda must be positive.');
    end
    if opts.gamma_w < 0
        error('RALMS:invalidGamma', 'gamma_w must be non-negative.');
    end
    if any(isnan(y)) || any(isnan(Phi(:)))
        error('RALMS:nanInput', 'Input contains NaN values.');
    end
    if any(~isfinite(y)) || any(~isfinite(Phi(:)))
        error('RALMS:nonFinite', 'Input contains non-finite values.');
    end

    tau     = opts.tau;
    gamma_w = opts.gamma_w;
    max_it  = opts.max_iter;
    tol_val = opts.tol;
    verbose = opts.verbose;

    % ======================== Lambda Selection =================================
    lam = opts.lambda;
    if isempty(lam)
        if verbose, fprintf('RALMS: Selecting lambda via BIC ...\n'); end
        grid = compute_lambda_grid_q(y, Phi, tau, 20, 1e-4);
        bics = zeros(length(grid), 1);
        for gi = 1:length(grid)
            [xh, ~, ~, ~] = ralms_solve(y, Phi, tau, grid(gi), ...
                                         gamma_w, max_it, tol_val, n, p_dim);
            bics(gi) = calculate_bic_q(y, Phi, xh, tau);
        end
        [~, best] = min(bics);
        lam = grid(best);
        if verbose, fprintf('  Selected lambda = %.6f\n', lam); end
    end

    % ======================== Main Estimation ==================================
    [x_hat, weights, n_iter, converged] = ...
        ralms_solve(y, Phi, tau, lam, gamma_w, max_it, tol_val, n, p_dim);

    result.x_hat     = x_hat;
    result.weights   = weights;
    result.lambda    = lam;
    result.tau       = tau;
    result.n_iter    = n_iter;
    result.converged = converged;

    % ======================== Print Summary ====================================
    if verbose
        print_ralms_result(result);
    end
end


% ====================== Core Solver ==========================================
function [x_hat, weights, n_iter, converged] = ...
        ralms_solve(y, Phi, tau, lam, gamma_w, max_iter, tol, n, p)

    % Adaptive weights from quantile Lasso initial estimator
    lam_init = 0.1 * sqrt(log(max(p, 2)) / max(n, 1));
    x_init = quantile_lasso_init(y, Phi, tau, lam_init);
    weights = 1 ./ (min(abs(x_init), abs(x_init - 1)) + 0.01).^gamma_w;

    L_Phi = estimate_L(Phi);
    x_hat = zeros(p, 1);
    lr = 1 / L_Phi;
    converged = false;
    n_iter = max_iter;

    for k = 1:max_iter
        x_old = x_hat;
        residuals = y - Phi * x_hat;
        subgrad = quantile_subgrad(residuals, tau);
        grad = -(Phi' * subgrad);
        v = x_hat - lr * grad;
        for j = 1:p
            x_hat(j) = prox_multi_signal(v(j), lr * lam * weights(j));
        end
        if norm(x_hat - x_old) < tol
            converged = true; n_iter = k; break;
        end
    end
end


% ====================== Helper Functions ======================================
function val = quantile_loss(u, tau)
    val = sum(u .* (tau - (u < 0)));
end

function g = quantile_subgrad(r, tau)
    g = zeros(size(r));
    g(r > 0) = tau;
    g(r < 0) = tau - 1;
    g(r == 0) = tau - 0.5;
end

function x_out = prox_multi_signal(v, lam_prime)
    % Proximal operator for penalty lambda * min(|x|, |x-1|)
    x1 = sign(v) * max(0, abs(v) - lam_prime);
    x2 = sign(v - 1) * max(0, abs(v - 1) - lam_prime) + 1;
    obj1 = 0.5*(x1 - v)^2 + lam_prime * min(abs(x1), abs(x1 - 1));
    obj2 = 0.5*(x2 - v)^2 + lam_prime * min(abs(x2), abs(x2 - 1));
    if obj1 <= obj2, x_out = x1; else, x_out = x2; end
end

function L = estimate_L(Phi, n_iter)
    if nargin < 2, n_iter = 30; end
    pp = size(Phi, 2);
    v = randn(pp, 1); v = v / norm(v);
    for i = 1:n_iter
        Av = Phi' * (Phi * v);
        nrm = norm(Av);
        if nrm < 1e-15, break; end
        v = Av / nrm;
    end
    L = max(v' * (Phi' * (Phi * v)), 1);
end

function x_hat = quantile_lasso_init(y, Phi, tau, lam)
    pp = size(Phi, 2);
    x_hat = zeros(pp, 1);
    L = estimate_L(Phi); lr = 1 / L;
    for k = 1:500
        x_old = x_hat;
        residuals = y - Phi * x_hat;
        grad = -(Phi' * quantile_subgrad(residuals, tau));
        v = x_hat - lr * grad;
        x_hat = sign(v) .* max(0, abs(v) - lr * lam);
        if norm(x_hat - x_old) < 1e-4, break; end
    end
end

function bic = calculate_bic_q(y, Phi, x_hat, tau)
    nn = length(y);
    residuals = y - Phi * x_hat;
    loss_val = 2 * quantile_loss(residuals, tau);
    k_active = sum(abs(x_hat) > 1e-4 & abs(x_hat - 1) > 1e-4);
    if k_active == 0, k_active = 0.5; end
    bic = nn * log(max(loss_val / nn, 1e-12)) + k_active * log(nn);
end

function grid = compute_lambda_grid_q(y, Phi, tau, n_grid, ratio)
    subgrad = quantile_subgrad(y, tau);
    lmax = max(abs(Phi' * subgrad));
    if lmax == 0, lmax = 1; end
    grid = exp(linspace(log(lmax), log(lmax * ratio), n_grid));
end

function print_ralms_result(r)
    fprintf('\n===== RALMS Estimation Result =====\n');
    fprintf('  tau (quantile level)   : %.4f\n', r.tau);
    fprintf('  lambda (regularization): %.6f\n', r.lambda);
    fprintf('  Iterations             : %d\n', r.n_iter);
    if r.converged, fprintf('  Converged              : Yes\n');
    else,           fprintf('  Converged              : No\n'); end
    pp = length(r.x_hat);
    n_zero = sum(abs(r.x_hat) < 1e-4);
    n_one  = sum(abs(r.x_hat - 1) < 1e-4);
    n_mid  = pp - n_zero - n_one;
    fprintf('  Signal dimension (p)   : %d\n', pp);
    fprintf('  Estimated edges (|x~1|): %d\n', n_one);
    fprintf('  Estimated zeros (|x~0|): %d\n', n_zero);
    fprintf('  Intermediate values    : %d\n', n_mid);
    if pp <= 30
        fprintf('  x_hat: ');
        fprintf('%.4f  ', r.x_hat);
        fprintf('\n');
    else
        fprintf('  x_hat (first 20): ');
        fprintf('%.4f  ', r.x_hat(1:20));
        fprintf('...\n');
    end
    fprintf('===================================\n');
end
