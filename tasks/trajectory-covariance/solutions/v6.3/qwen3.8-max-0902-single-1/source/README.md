# Long-run covariance from a single MCMC trajectory

`./run.sh INPUT.npy OUTPUT.npy` reads an `(M, d)` float64 trajectory (rows in time
order) and writes the `(d, d)` long-run covariance `H`, i.e. an estimate of
`Sigma = lim n Cov(mean of n steps)`, as C-order little-endian float64.

## Dependencies
Python 3 + NumPy only (tested with Python 3.11, NumPy 1.26). No build step is
needed; `build.sh` merely checks the imports and pre-compiles the module.
Runtime is ~0.4 s for M = 80000, d = 4 (single thread, far below the 5 s limit).

## Method
1. All autocovariance matrices `G_k = (1/M) sum_t y_t y_{t+k}^T`, `k = 0..M-1`
   (`y` = mean-centred data), computed with one zero-padded FFT per component and
   `d(d+1)/2` inverse FFTs.
2. A pilot estimate `H0` is formed with a Bartlett window whose bandwidth comes
   from an initial-positive-sequence truncation of the averaged autocorrelation
   `tr(M G_k)/tr(M G_0)` (`M = Gamma_0^{-1}`), giving a metric proxy `M = H0^{-1}`
   for `Sigma^{-1}` and an average integrated autocorrelation time `tau`.
3. The delivered estimator is the lag-window estimator `H(b) = sum_k w(k/b) G_k`
   with the Bartlett kernel. The bandwidth minimises an estimate of the scored
   loss `E ||S^{-1/2}(H-S)S^{-1/2}||_F^2`, split into
   * squared bias `~ c_q^2 b^{-2q} ||D_q||^2` with `D_q = sum_k |k|^q G_k`, and
   * total variance `~ (b/M) * int w^2 * c_v` with
     `c_v = sum_m [(tr G_m)^2 + ||G_m||_F^2]`,
   both evaluated in the whitened metric `M` (which is what the score measures)
   and both debiased for the sampling noise of the empirical `G_k`. This yields
   `b* = [2 q c_q^2 ||D_q||^2 M / (int w^2 c_v)]^{1/(2q+1)}`, clamped to
   `[max(4, tau/2), min(M/4, max(10, 6 tau))]` for robustness.
4. The output is symmetrised and projected onto the PSD cone (negative
   eigenvalues clipped to zero) before being written.

Everything is affine-equivariant: a linear map / shift of the input columns maps
the output by `A^T H A`, so the estimate is unaffected by the per-trajectory
reparametrisation mentioned in the task. No randomness is used, so `SEED` is
irrelevant; no reference values, tables or model identification are used — the
bandwidth and all tuning quantities are estimated from the given trajectory.

## Files
* `lrcov.py` – estimator (also exposes comparator estimators used in development).
* `main.py` – CLI entry point used by `run.sh`.
* `run.sh` / `build.sh` – runner and (trivial) build script.
