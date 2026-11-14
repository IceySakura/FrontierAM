# Long-run covariance estimator from one trajectory

## What the program does

`run.sh INPUT.npy OUTPUT.npy` reads a trajectory `X` of shape `(M, d)` (rows are
sampling times) and writes a single `d x d` array `H` estimating

    Sigma = lim_{n->inf} n Cov(mean(X_1..X_n)) = sum_{k=-inf}^{inf} Gamma(k),

i.e. `2 pi` times the spectral density matrix at frequency zero.  The matrix
`H` is finite, symmetric, positive semidefinite, C-order and little-endian
float64.

## Method

The estimator is affine equivariant: replacing the input by `X A + b` changes
the output to `A^T H A`.  This matters because a trajectory may be supplied in
linearly transformed coordinates.

1. **Whitening.** Center the data and whiten it with the symmetric inverse
   square root of the lag-zero covariance `C0 = Xc^T Xc / M`.  All later
   operations are functions of whitened covariance matrices and therefore
   equivariant.

2. **VAR prewhitening.** Fit vector autoregressions of every order `0..P`
   (`P = 8`) by least squares using one shared Gram matrix, and choose an order
   by AIC.  The filter is *not* assumed to be the true model: for any stable
   invertible linear filter `e_t = x_t - sum_i x_{t-i} A_i` the exact identity

       Sigma_x = B^{-T} Sigma_e B^{-1},       B = I - sum_i A_i,

   holds, where `Sigma_e` is the long-run covariance of the filtered series.
   Prewhitening makes the residual spectrum nearly flat, so `Sigma_e` is much
   easier to estimate than `Sigma_x`.

3. **Residual correction.** Estimate `Sigma_e` with a flat-top lag-window
   estimator whose bandwidth is chosen by a Politis-style adaptive rule (the
   first block of consecutive lags whose whitened autocorrelation matrices are
   all below a noise-level threshold estimated from far lags).  A James--Stein
   factor derived from the estimated variance of the correction shrinks it
   toward white noise, so no variance is added when the residuals are white.
   The residual covariance uses a degrees-of-freedom correction for the fitted
   VAR coefficients.

4. **Recolor and project.** Apply the identity above, symmetrize, and project
   onto the positive semidefinite cone (eigenvalue clipping).

5. **Mode-dependent Stein shrinkage.** In lag-zero-whitened coordinates the
   eigenvalues `tau_i` of `W = G^{-1/2} H G^{-1/2}` are the integrated
   autocorrelation times of the modes; mode `i` has about `nu_i = M / tau_i`
   effectively independent samples.  Under the whitened Frobenius loss the
   best multiple of a noisy covariance estimate is `nu/(nu + d + 1)`, so mode
   `i` is shrunk by `nu_i/(nu_i + (d+1))`.  Slow modes are regularized
   strongly, nearly independent modes are left untouched.

6. **Shrinkage toward a common autocorrelation shape.**  If all coordinates
   share one autocorrelation shape, `S = tau * G` with `G` the lag-zero
   covariance, and `tau*G` is a much lower-variance estimate than the
   unrestricted matrix.  A second James--Stein weight, computed from the
   `H`-whitened discrepancy between `H` and `tau*G` and an estimate of the
   sampling variance, decides how much of that structured estimate to use.
   It is close to one when the common-shape model is supported by the data and
   close to zero otherwise.  This step is also affine equivariant.

7. **Safety.** All eigen-decompositions use eigenvalue floors; every matrix is
   symmetrized and projected onto the PSD cone.  A cheap Bartlett-kernel
   fallback is used if anything unexpected fails, and non-finite or malformed
   inputs are sanitized, so a valid `d x d` matrix is always produced.

## Dependencies and build

Python 3 with NumPy only (no SciPy, no compiled extensions, no network).
There is nothing to build, so no `build.sh` is needed.  `run.sh` is executable
and uses the `python3` on `PATH`; it restricts BLAS to one thread.

## Files

- `lrc.py` — complete source and the command-line entry point.
- `run.sh` — executable entry point: `./run.sh INPUT OUTPUT`.
- `README.md` — this document.

The program is deterministic (no randomness, no use of `SEED`), reads only the
given input file, and performs one FFT-based autocovariance pass plus a handful
of small matrix factorisations.  The largest trajectories (80,000 x 6) take
well under one second on a single CPU, far below the five-second limit.
