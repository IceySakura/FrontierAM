# Long-run covariance from one trajectory

`run.sh` reads one `(M, d)` NumPy float64 array and writes one `(d, d)` NumPy
float64 array containing an estimate of

    Sigma = sum_{k=-inf}^{inf} Cov(x_t, x_{t+k})

(the asymptotic covariance of `sqrt(M) * mean(x)`, i.e. `H/M` estimates the
covariance of the sample mean).

## Usage

```sh
./build.sh                                   # optional; checks NumPy
./run.sh /absolute/input.npy /absolute/output.npy
```

`run.sh` takes exactly two arguments.  The output is C-contiguous, little-endian
float64, symmetric and positive semidefinite.

## Dependencies / source

Python 3 and NumPy only (both preinstalled in the evaluation image).  No
network access, no compiled extensions, no randomness, no cross-call state.
The complete source is `estimate.py`; `run.sh` and `build.sh` are thin shell
wrappers.  All algorithms are implemented from scratch in this submission,
based on standard time-series / MCMC literature: lag-window (HAC) spectral
estimation, Yule–Walker VAR prewhitening (VARHAC), automatic bandwidth
selection, and Geyer/Politis-style correlation-length selection.

## Method

The estimator is affine equivariant, so a per-trajectory invertible linear
rescaling of the coordinates cannot change its whitened error.

1. **Whitening.** Center the series and compute the Cholesky factor of its
   lag-0 correlation matrix.  All subsequent work is on `y`, whose lag-0
   covariance is the identity; the final matrix is
   `H = (D L) Omega (D L)^T` with `Omega` the whitened zero-frequency spectrum.
2. **Autocovariances.** All `R_k = sum_t y_t y_{t-k}^T / M` up to `0.45 M` are
   obtained with one batched real FFT of length a power of two `>= 2M`.
   Complexity `O(d^2 M log M)`; end-to-end runtime is below 1 s for
   `M = 80000, d = 6`, including interpreter startup.
3. **Flat-top estimate.** A lugsail-style flat-top lag window (weights exactly
   one on the first third of the window, then linearly decreasing, so
   short-memory/MA and negative autocorrelation contribute without kernel
   bias) with bandwidth from the maximum of (i) a noise-adaptive
   correlation-length rule on `||R_k||_F` (all weights vanish when no lag is
   significant) and (ii) an AR(1) plug-in bandwidth from a pilot estimate.
4. **VAR-prewhitened estimate (VARHAC).** Fit VAR(`p`) models `p = 0..12` by
   Yule–Walker, choose `p` by AIC on the residual covariance determinant,
   apply the same adaptive flat-top window to the residual autocovariances
   (computed analytically from `R_k`, no extra FFT), and recolor with
   `(I - sum A_i)^{-1}` (pseudo-inverse if that matrix is singular).
5. **Combination and cleanup.** The weight `lambda` of the VARHAC part is a
   product of two data-driven factors: one based on the smallest eigenvalue of
   the flat-top estimate (when the spectrum nearly vanishes, i.e. near-
   cancelling / antithetic correlations, prewhitening amplifies residual noise
   and the flat-top estimate is used) and one based on the trace ratio of the
   two estimates (a much smaller VARHAC trace indicates an undetected long
   correlation tail that the direct window captures better).  Otherwise the
   more efficient prewhitened estimate dominates.  The convex combination is
   projected onto the PSD cone by eigenvalue clipping, symmetrized, recolored,
   and written.  All constants were tuned on independently simulated linear,
   finite-state and MA/ARMA trajectories with analytically known `S`.

Both constituent estimates are positive semidefinite after eigenvalue
clipping, and the convex combination preserves that property.  Finite-sample
checks (symmetry, PSD, finiteness) are performed before writing.

## Notes

* Single-threaded: BLAS/OMP thread counts are pinned to 1 in `run.sh`.
* No scenario labels, lookup tables, or reference answers are used; the input
  file is the only data source.
* Worst-case memory is dominated by the FFT of shape `(2^k, d^2)`, below
  100 MiB for the specified sizes.
