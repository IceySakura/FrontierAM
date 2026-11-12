# Long-run covariance from one trajectory

`./run.sh /abs/input.npy /abs/pred.npy` reads an `(M, d)` float64 trajectory and writes
the `(d, d)` long-run covariance estimate `H` (symmetric PSD, C-order little-endian float64).

## Dependencies
Python 3 and NumPy only (tested with Python 3.11, NumPy 1.26). No build step, no network,
single thread (BLAS threads are pinned to 1). Randomness (posterior draws) uses the
environment variable `SEED` (default 1), so results are deterministic.

## Method (`lrcov.py`)
1. **Whitening**: centre the data and whiten with the sample covariance. Every step is
   affine equivariant, so a fixed invertible linear transform and shift of the input
   coordinates changes `H` exactly as `A H A'`.
2. **Finite-state chains**: if the trajectory takes at most `min(1500, M/10)` distinct
   values that are each re-entered repeatedly (median entries >= 3, which excludes
   Metropolis rejections), estimate the transition matrix from the transition counts
   (closed with one wrap-around transition) and use the exact Markov-chain formula
   `Sigma = F'(D Z + Z' D - D) F`, `Z = (I - P + 1 pi')^{-1}`.
3. **Otherwise VAR(p)**: least-squares VAR fits with orders 0..pmax (`pmax ~ 24 (M/1e4)^{1/4}`),
   order chosen by AIC, long-run covariance `A(1)^{-1} Sigma_u A(1)^{-T}`.
4. **Risk-minimising adjustment**: 200 draws from the posterior of the fitted model
   (normal/inverse-Wishart posterior of the VAR under a Jeffreys prior; Dirichlet rows of
   the transition matrix for finite chains, when <= 150 states). The output minimises the
   average whitened Frobenius error `||S_b^{-1/2}(H - S_b)S_b^{-1/2}||_F` over the draws
   (IRLS / Weiszfeld). This shrinks noisy, right-skewed estimates of strongly autocorrelated
   chains and changes almost nothing when the estimate is precise.
5. Fallbacks (plain VAR, batch means, sample covariance) guarantee a valid PSD output.

## Sources
Standard textbook methods: VAR spectral density at frequency zero (e.g. Lütkepohl,
*New Introduction to Multiple Time Series Analysis*), the Markov chain fundamental matrix
(Kemeny & Snell), and Bayesian multivariate regression posteriors. All code is original.

## Files
- `run.sh` – entry point (calls `run_lrcov.py`).
- `run_lrcov.py` – I/O, validation, fallbacks.
- `lrcov.py` – the estimator.
