% ==============================================================================
% Test Script for RALMS and CRALMS (MATLAB version)
% ==============================================================================
% Verifies that both RALMS() and CRALMS() functions work correctly on
% synthetic data with known ground truth.
% ==============================================================================

fprintf('======================================================\n');
fprintf('  Test Suite for RALMS and CRALMS (MATLAB version)\n');
fprintf('======================================================\n\n');

rng(42);
n_pass = 0;
n_fail = 0;

% ==============================================================================
% Test 1: Basic functionality with simple synthetic data
% ==============================================================================
fprintf('--- Test 1: Basic estimation on synthetic data ---\n');

N = 8; p = N*(N-1)/2;  % p = 28
n = N * 10;             % 80

x_true = zeros(p, 1);
idx = randperm(p, round(0.3 * p));
x_true(idx) = 1;

Phi = randn(n, p);
y = Phi * x_true + randn(n, 1) * 0.3;

% RALMS with auto-lambda
fit_r = RALMS(y, Phi, 'tau', 0.5, 'verbose', true);
[n_pass, n_fail] = report('RALMS returns struct',   isstruct(fit_r), n_pass, n_fail);
[n_pass, n_fail] = report('RALMS x_hat length',     length(fit_r.x_hat) == p, n_pass, n_fail);
[n_pass, n_fail] = report('RALMS lambda positive',  fit_r.lambda > 0, n_pass, n_fail);

mse_r = mean((x_true - fit_r.x_hat).^2);
fprintf('  RALMS MSE = %.6f\n', mse_r);
[n_pass, n_fail] = report('RALMS converged or accurate', fit_r.converged || mse_r < 0.01, n_pass, n_fail);
[n_pass, n_fail] = report('RALMS MSE reasonable', mse_r < 0.5, n_pass, n_fail);

% CRALMS with auto-lambda
fit_c = CRALMS(y, Phi, 'tau', 0.5, 'verbose', true);
[n_pass, n_fail] = report('CRALMS returns struct',  isstruct(fit_c), n_pass, n_fail);
[n_pass, n_fail] = report('CRALMS x_hat length',    length(fit_c.x_hat) == p, n_pass, n_fail);
[n_pass, n_fail] = report('CRALMS x_hat in [0,1]',  all(fit_c.x_hat >= -1e-10) && all(fit_c.x_hat <= 1+1e-10), n_pass, n_fail);
[n_pass, n_fail] = report('CRALMS lambda positive', fit_c.lambda > 0, n_pass, n_fail);

mse_c = mean((x_true - fit_c.x_hat).^2);
fprintf('  CRALMS MSE = %.6f\n', mse_c);
[n_pass, n_fail] = report('CRALMS MSE reasonable', mse_c < 0.5, n_pass, n_fail);

% ==============================================================================
% Test 2: User-supplied lambda
% ==============================================================================
fprintf('\n--- Test 2: User-supplied lambda ---\n');

fit_r2 = RALMS(y, Phi, 'tau', 0.5, 'lambda', 0.05);
[n_pass, n_fail] = report('RALMS user-lambda stored', abs(fit_r2.lambda - 0.05) < 1e-10, n_pass, n_fail);

fit_c2 = CRALMS(y, Phi, 'tau', 0.5, 'lambda', 0.05);
[n_pass, n_fail] = report('CRALMS user-lambda stored', abs(fit_c2.lambda - 0.05) < 1e-10, n_pass, n_fail);

% ==============================================================================
% Test 3: Heavy-tailed noise (t-distribution, df=2)
% ==============================================================================
fprintf('\n--- Test 3: Heavy-tailed noise (t2) ---\n');

% MATLAB trnd requires Statistics Toolbox; use randn-based approximation if unavailable
try
    noise_t2 = trnd(2, n, 1) * 0.5;
catch
    % Fallback: ratio of normals approximation
    noise_t2 = randn(n,1) ./ sqrt(chi2rnd(2, n, 1) / 2) * 0.5;
end
y_heavy = Phi * x_true + noise_t2;

fit_r3 = RALMS(y_heavy, Phi, 'tau', 0.5, 'lambda', 0.1);
fit_c3 = CRALMS(y_heavy, Phi, 'tau', 0.5, 'lambda', 0.1);

mse_r3 = mean((x_true - fit_r3.x_hat).^2);
mse_c3 = mean((x_true - fit_c3.x_hat).^2);
fprintf('  RALMS  MSE (t2 noise) = %.6f\n', mse_r3);
fprintf('  CRALMS MSE (t2 noise) = %.6f\n', mse_c3);
[n_pass, n_fail] = report('RALMS handles t2 noise',  isfinite(mse_r3), n_pass, n_fail);
[n_pass, n_fail] = report('CRALMS handles t2 noise', isfinite(mse_c3), n_pass, n_fail);

% ==============================================================================
% Test 4: Different tau values
% ==============================================================================
fprintf('\n--- Test 4: Non-default tau ---\n');

fit_r4 = RALMS(y, Phi, 'tau', 0.25, 'lambda', 0.1);
[n_pass, n_fail] = report('RALMS tau=0.25 works', fit_r4.tau == 0.25, n_pass, n_fail);

fit_c4 = CRALMS(y, Phi, 'tau', 0.75, 'lambda', 0.1);
[n_pass, n_fail] = report('CRALMS tau=0.75 works', fit_c4.tau == 0.75, n_pass, n_fail);

% ==============================================================================
% Test 5: Input validation (should produce errors)
% ==============================================================================
fprintf('\n--- Test 5: Input validation ---\n');

% Dimension mismatch
try
    RALMS(y(1:5), Phi);
    [n_pass, n_fail] = report('RALMS dim mismatch error', false, n_pass, n_fail);
catch e
    [n_pass, n_fail] = report('RALMS dim mismatch error', ...
        contains(lower(e.message), 'mismatch'), n_pass, n_fail);
end

% Invalid tau
try
    RALMS(y, Phi, 'tau', 1.5);
    [n_pass, n_fail] = report('RALMS invalid tau error', false, n_pass, n_fail);
catch e
    [n_pass, n_fail] = report('RALMS invalid tau error', ...
        contains(lower(e.message), 'tau'), n_pass, n_fail);
end

% Negative lambda
try
    CRALMS(y, Phi, 'lambda', -1);
    [n_pass, n_fail] = report('CRALMS negative lambda error', false, n_pass, n_fail);
catch e
    [n_pass, n_fail] = report('CRALMS negative lambda error', ...
        contains(lower(e.message), 'lambda'), n_pass, n_fail);
end

% NaN in y
y_nan = y; y_nan(1) = NaN;
try
    RALMS(y_nan, Phi);
    [n_pass, n_fail] = report('RALMS NaN check', false, n_pass, n_fail);
catch e
    [n_pass, n_fail] = report('RALMS NaN check', ...
        contains(lower(e.message), 'nan'), n_pass, n_fail);
end

% ==============================================================================
% Test 6: Small problem
% ==============================================================================
fprintf('\n--- Test 6: Small problem (n=10, p=3) ---\n');

Phi_small = randn(10, 3);
x_small = [1; 0; 1];
y_small = Phi_small * x_small + randn(10, 1) * 0.1;

fit_rs = RALMS(y_small, Phi_small, 'lambda', 0.01);
fit_cs = CRALMS(y_small, Phi_small, 'lambda', 0.01);
[n_pass, n_fail] = report('RALMS small problem',  length(fit_rs.x_hat) == 3, n_pass, n_fail);
[n_pass, n_fail] = report('CRALMS small problem', length(fit_cs.x_hat) == 3, n_pass, n_fail);
fprintf('  x_true  = [%.3f, %.3f, %.3f]\n', x_small);
fprintf('  RALMS   = [%.3f, %.3f, %.3f]\n', fit_rs.x_hat);
fprintf('  CRALMS  = [%.3f, %.3f, %.3f]\n', fit_cs.x_hat);

% ==============================================================================
% Summary
% ==============================================================================
fprintf('\n======================================================\n');
fprintf('  Results: %d passed, %d failed, %d total\n', n_pass, n_fail, n_pass + n_fail);
if n_fail == 0
    fprintf('  ALL TESTS PASSED\n');
else
    fprintf('  SOME TESTS FAILED — please review above\n');
end
fprintf('======================================================\n');


% ====================== Helper function =======================================
function [np, nf] = report(test_name, passed, np, nf)
    if passed
        fprintf('  [PASS] %s\n', test_name);
        np = np + 1;
    else
        fprintf('  [FAIL] %s\n', test_name);
        nf = nf + 1;
    end
end
