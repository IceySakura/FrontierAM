# 3-D fluorescence deconvolution (fixed CPU budget)

`python3 -I -B solution.py INPUT.npz OUTPUT.npy`. Pure numpy; no build step,
no weights, no other files. Typical wall time per 32^3 case is about 0.17 s
including interpreter start-up and imports; the dev core is shared, so this is
only approximate.

## Method

* **Exact forward model.** `y = crop(conv(zero-extend(x), psf))`. When the PSF is
  separable (rank-1 within float32 rounding, which is true of every supplied PSF),
  each axis becomes a banded 32x32 "same" matrix with the zero extension and crop
  built in. The full 3-D operator is then three small matmuls, about 0.25 ms. A
  non-separable PSF falls back to `rfftn` on a padded grid that can't alias.
* **Prior.** Synthesis model `x = G c`, where `c >= 0` is sparse (L1) and `G` is an
  isotropic Gaussian blob (sigma 0.8 voxel, radius 3). This matches the
  point-blob and tube cross-sections. `G` wraps periodically, which couples
  opposite faces. That noticeably helps sheet and tube structures that cross the
  volume ("boundary-adjacent" cases) and made no difference on the other public
  cases.
* **Noise handling.** The read-noise sigma `s` is estimated from the darkest 30%
  of voxels (by a doubly blurred copy), using `E[y^2] = s^2/2` for a zero-mean
  Gaussian clipped at 0. Observed zeros are censored, so they are replaced by the
  conditional mean of the censored noise, `-0.798 s`. This removes the positive
  bias that clipping causes.
* **Solver.** FISTA with a nonnegative soft-threshold step. Step size `1/||A||^2`
  comes from the exact spectral norms of the per-axis matrices. The L1 weight and
  the early-stopping count are interpolated linearly in `s` between the two noise
  regimes seen in the dev data: about 0.003 and 60 iterations at `s ~ 0.008`,
  about 0.0065 and 45 iterations at `s ~ 0.025`.
* **Debiasing.** L1 shrinks amplitudes, so one global amplitude is refit by least
  squares to the data, clipped to [0.9, 1.5].
* **Safety.** The run respects `DECONV_DEADLINE_MONOTONIC` and
  `TIME_BUDGET_SECONDS`, capped at 0.5 s and minus a 0.13 s margin. The iteration
  loop stops early if time runs short. Any solver exception falls back to a
  valid nonnegative float32 output. BLAS is pinned to one thread.

## Development results (public-dev, 12 cases, mean nMSE)

| method | mean nMSE |
| --- | --- |
| Richardson-Lucy, best fixed iteration count (20) | 0.222 |
| this method | 0.081 |

## Sources and dependencies

All code is original to this submission. It uses only numpy 1.26 (pre-installed)
and the Python standard library. Parameters were tuned on the twelve public
development cases only. No external data or weights are used.
