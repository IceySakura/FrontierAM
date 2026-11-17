# 3D Fluorescence Deconvolution — submission

Single-file solver: `solution.py` (numpy only, no build step, no data files, no network).

    python3 -I -B solution.py /abs/raw.npz /abs/pred.npy

`build.sh` is intentionally absent: nothing has to be compiled or downloaded.

## Dependencies and sources

* **Python 3 stdlib** (`os`, `sys`, `time`) and **numpy** (1.26.4 in the image) — the
  only imports. scipy/torch are *not* imported: their import cost alone (0.36 s / 1.3 s)
  would exceed the per-case budget.
* No offline weights, no cached answers, no external files. All tuning constants were
  derived by the author from the twelve public dev examples (`public-dev/case_*.npz`
  and their truths) and are hard-coded as four numbers at the top of `solution.py`.
* All code is original, written for this task.

## Method

Forward model of the task: `observed = clip(Poisson(s·(P·x))/s + N(0, sg²), 0)` with `P`
= centred `same` 3D convolution with the supplied PSF, zero-extended outside the grid.

1. **Blind noise calibration.** The PSF is a (truncated) Gaussian, so its transfer
   function is negligible above ~0.4 of the Nyquist radius; the mean power of `observed`
   in that shell is white-noise power (read noise + Poisson + rectification).
   `sig = sqrt(power)` is scale-equivariant, so the same constants adapt to both noise
   regimes and to any overall intensity scale. Nothing else about the case is assumed.
2. **Richardson–Lucy / EM with a spurious background** `c = 3·sig`. Adding a constant
   background of order `s·sg²` makes the Poisson variance `(f+c)/s` reproduce the true
   variance `f/s + sg²` (the standard Poisson–Gaussian "MA-RL" trick) and keeps the
   ratio `(y+c)/(P x+c)` well defined in the regions clipped to zero.
3. **L1 / exponential prior**, applied as a constant `lam = 0.05` added to the
   sensitivity image `P^T 1` in the multiplicative update. It contracts the dark
   background geometrically towards zero, which matters because the clean volume is
   exactly zero over most voxels.
4. **Early stopping** (`N_ITER = 30`) is the dominant regulariser: the run stops long
   before the heavily biased MAP fixed point.
5. **Final soft threshold** at `0.5·sig` removes the residual noise haze that nMSE
   punishes in dark regions.

Public-dev result (12 cases, mean nMSE vs. the provided truths): **0.1197**
(raw observation as the answer: 0.31–1.01 per case, mean ≈ 0.49).

## Operator and speed

The PSFs of this family are exactly rank-1 separable (verified to ~1e-7 relative).
`Op` detects separability from the PSF marginals and, when it holds, evaluates `P` and
`P^T` as three banded BLAS matmuls per axis in float32 (~0.30 ms / 0.33 ms per
application) instead of a 40³ rfft pair (~2.3 ms). The operator was validated against
`scipy.signal.fftconvolve(..., mode='same')` (max relative error 1.8e-7, i.e. float32
rounding) and with an adjoint dot-product test (relative error ~1e-10). General,
non-separable PSFs fall back to the exact padded-rfft path.

Whole-process wall clock measured with `python3 -I -B solution.py …` on the dev cases:
**0.135–0.147 s** (interpreter start + numpy import ≈ 0.09 s, load, compute, write,
exit). Memory is a few MB.

## Deadline safety

* `DECONV_DEADLINE_MONOTONIC` (if present and sane) is honoured, minus a 50 ms write
  reserve, and is additionally capped at 0.40 s after process start — so the solver stays
  inside the 0.5 s budget even if the harness clock differs, and far inside 1.25 s.
* The iteration loop re-measures its own per-iteration cost every ~12 ms and stops early
  if the remaining iterations cannot fit (this is what protects the slower rfft fallback
  path on unexpected inputs).
* A valid float32 array (the clipped observation) is written and `fsync`ed **before** the
  iteration starts, then atomically replaced via `os.replace` of a temp file, so an
  output always exists.
* Everything is wrapped: on any exception the handler writes a finite nonnegative array
  of the input shape and exits 0. Output is forced finite, nonnegative, float32, shape
  `(Z,Y,X)`, saved with `allow_pickle=False`.
* No CPU index is hard-coded; BLAS/OMP thread variables are pinned to 1 and the
  inherited affinity is left untouched.
