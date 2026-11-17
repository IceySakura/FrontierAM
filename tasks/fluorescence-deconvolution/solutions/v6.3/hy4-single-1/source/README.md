# 3-D fluorescence deconvolution: submission notes

## Files

* `solution.py` — the solver.  Invoked as
  `python3 -I -B solution.py INPUT.npz OUTPUT.npy`.
* `model.npz` — offline-fitted coefficients of the linear reconstruction
  stage (`mu`, `sd`, `w`; 65 features plus an intercept).  It is a plain
  NumPy `.npz` of float32 arrays, loaded with `allow_pickle=False`; it
  contains no executable code.
* No `build.sh` is required; everything is shipped pre-built.

## Dependencies

Python 3.11, NumPy 1.26.4 and SciPy 1.12.0 (only `scipy.fft` and
`scipy.ndimage`).  All are part of the supplied offline environment.  The
solver forces single-threaded BLAS/FFT through environment variables set
before NumPy is imported, and passes `workers=1` to every FFT.  No network,
GPU, compiler, or extra packages are used.  Model weights were fitted offline
by ridge-weighted least squares on the twelve public development volumes in
`public-dev/` plus 60 synthetic variants of those same public volumes (random
separable Gaussian PSFs with per-axis widths in [0.9, 2.2] and fresh draws of
the two observed noise regimes).  Only public data were used; no hidden data,
private generator, evaluator output, or cross-case state is used.  The source
of every weight is therefore the supplied `public-dev/` examples.

## Method

1. **Noise regime.**  A robust MAD of `y` minus the mean of its six nearest
   neighbours estimates the additive Gaussian width.  The two noise regimes
   of this task are cleanly separated, so the estimate is snapped to
   `sigma = 0.008` or `0.025`.  A baseline of `0.4*sigma` (the mean of the
   clipped Gaussian noise) is subtracted from the observation.

2. **Forward model.**  The PSF-centered zero-extension `same` convolution A
   and its *exact* adjoint Aᵀ are evaluated with real FFTs on a grid of size
   `N + P - 1` (40³ for the formal 32³ / 9³ inputs).  The kernel is rolled so
   that the centre voxel sits at the origin; the adjoint uses `conj(H)`.
   This handles boundary-adjacent objects without wrap-around artefacts.

3. **Richardson–Lucy bank.**  Forty baseline-corrected RL iterations are run;
   the multiplicative update provides a diagonal-in-space preconditioner that
   converges quickly on sparse non-negative objects.  Twelve iterates
   (`k = 4 … 40`) are retained.

4. **Features (65 per voxel).**  The retained iterates; 3³ local means and
   maxima of four iterates; 5³ local means; per-axis 3-tap local means (which
   let the model cancel anisotropic sidelobes); rectified iterates at five
   thresholds scaled by `sigma`; the matched-filter map Aᵀy and three
   residual-correlation maps Aᵀ(y − A x_k); a square-root and a ratio
   feature; and four spectrally regularized Wiener inverses plus their local
   means and rectified versions.

5. **Reconstruction.**  A single linear model (ridge-regularized weighted
   least squares, fitted on the public volumes and their synthetic PSF
   variants with per-case weights `1/mean(truth²)`) maps the standardized
   features to the estimate, which is finally clipped at zero.  Combining
   iterates removes the bias/variance trade-off of any single RL stopping
   time; leave-one-case-out mean nMSE on the public set is ≈ 0.077, versus
   ≈ 0.15 for the best fixed RL stopping rule.

## Runtime and safety

A fallback reconstruction (the tenth RL iterate) is written to the output
path before any long computation, so a valid `.npy` always exists.  The
remaining RL iterations and the model stage run only while time remains
before `DECONV_DEADLINE_MONOTONIC` (or `TIME_BUDGET_SECONDS`), with a
reserved exit margin; the final estimate overwrites the fallback only when
the full stage completes.  Measured end-to-end time including interpreter
startup, imports, input, compute and output is ≈ 0.34 s per case on the
development host, leaving a wide margin under the wall-clock limit.  Memory
use is well below 100 MiB.  Output is a C-contiguous `float32` `.npy` array
of the same shape as `observed`, finite and non-negative.

## Validation

`python3 -I -B solution.py INPUT OUTPUT` followed by
`python3 public-dev/check.py public-dev/case_01.npz
public-dev/case_01_truth.npy pred.npy` was run during development
(`mse=0.000147316159 nmse=0.0184173465`); the twelve public examples give
mean nMSE ≈ 0.073.  Additional development checks covered unseen PSF sizes
(7³ and 11³), asymmetric PSFs, non-32³ volumes, intermediate noise levels
and artificially tight deadlines; all produced finite non-negative float32
outputs with the required shape.
