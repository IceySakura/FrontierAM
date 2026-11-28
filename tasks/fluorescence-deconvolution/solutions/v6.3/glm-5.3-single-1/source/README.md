# Three-Dimensional Fluorescence Deconvolution

## What it does

`solution.py` recovers a non-negative 32×32×32 intensity volume from a noisy,
truncated, blurred observation and its 9×9×9 point-spread function.

```
python3 -I -B solution.py /absolute/path/raw.npz /absolute/path/pred.npy
```

The only dependency is **numpy** (already installed in the image). There are no
shipped weights, no network access, no build step (no `build.sh` is needed) and
no other files are required.

## Forward model

The observation is a zero-extended `'same'` 3-D linear convolution of the clean
volume with the (centered, odd, unit-sum) PSF, followed by Poisson shot noise,
additive Gaussian read noise and truncation at zero:

```
y = max(A x + n, 0),   Var(n_i) = sigma^2 + (A x)_i / gain
```

`A` is exactly the operator described in the statement (zero extension outside
the clean grid, central crop of the full linear convolution). `A^T` is the
matched adjoint, i.e. the same convolution with the flipped PSF; this matters
because the finite support makes the operator non-shift-invariant near the
faces of the cube.

## Estimation

The solver minimises a **censored-Gaussian** data term plus a smoothed
total-variation prior over the non-negative orthant:

```
min_{x >= 0}  sum_i rho_i((A x)_i)  +  lam * TV_eps(x)

rho_i(b) = (b - y_i)^2 / (2 v_i)            if y_i > 0
           -log Phi(-b / sqrt(v_i))          if y_i == 0      (censored voxel)
v_i      = sigma_w^2 + b / gain
```

Two parts of this are what actually buy accuracy:

* **The censoring term.** In low-signal regions roughly half of the voxels are
  truncated to exactly zero. Treating them as ordinary Gaussian residuals
  forces a large spurious background (the rectified noise floor alone carries
  more total intensity than a sparse sample). The `-log Phi(-b/s)` term uses
  the information "the untruncated value was non-positive" through the inverse
  Mills ratio, which cancels that bias: recovered totals match the blurred
  totals to a few percent.
* **Noise-proportional weights.** `sigma_w` is the estimated read noise
  inflated by 2×, so noise-dominated voxels contribute little. This is a
  robustness device: with an exact read-noise scale the dark regions dominate
  the gradient and the reconstruction tracks the noise floor.

The prior is isotropic 3-D TV smoothed with `eps = 0.08` (Huber-like, so tubes
keep their edges without staircasing) with `lam = 0.5`; both were tuned on the
twelve public development examples by mean normalised squared error.

The optimiser is projected **FISTA** (accelerated projected gradient) with an
explicit Lipschitz step. The iteration count is the remaining regulariser: it
is bounded by the wall-clock budget, and the process always leaves enough time
to convert, write and close the output.

## Per-case parameters, derived from the observation only

* **Read-noise scale `sigma`.** For a voxel whose blurred intensity is
  negligible the observation is `max(n, 0)` with `n ~ N(0, sigma^2)`; its CDF
  at `t > 0` is `Phi(t/sigma)` while `P(y == 0) = 1/2`. Matching the empirical
  low quantiles of the positive values against the fraction of exact zeros
  recovers `sigma` with no knowledge of the clean volume (median of five
  quantile matches). On the public examples this lands within ~5 % of the true
  read noise for both noise regimes.
* **Shot-noise gain.** Inferred from the read-noise scale via the relation the
  two noise regimes follow, and clamped to a sane range. The reconstruction is
  insensitive to moderate errors here (±50 % in the gain changes the mean
  normalised error by ~0.003).
* **Stopping.** The loop is bounded by `min(own budget, deadline, time budget)`
  where the own budget is 0.40 s measured from process launch (which covers
  interpreter start-up and imports) with a reserve for writing the output.

## Numerical kernels

Both `A` and `A^T` are implemented twice:

* a **separable fast path** — the PSFs are outer products of three 1-D kernels
  (verified at run time by reconstructing the PSF from its axis marginals; the
  accepted relative error is < 1e-3), so the 3-D convolution is three 9-tap
  1-D passes in float32, ~2.5× faster than an FFT round trip;
* a **general FFT path** — zero-padded to `(n + m - 1)` per axis, used for any
  PSF that is not separable, so correctness never depends on the separability
  assumption.

## Safety

The whole computation is wrapped so that any failure falls back to emitting
the (already non-negative, finite, correctly shaped) observation rather than
an invalid array. The estimate is clipped to zero and checked for finiteness
before `np.save(..., allow_pickle=False)`.

## Result on the public development examples

Mean normalised squared error over the twelve fixed examples: **0.127**
(the observation itself scores 0.498, a flat field at the observed mean 0.97,
and plain Richardson–Lucy at its best fixed iteration count 0.19). Worst-case
process wall clock on the development core: 0.38 s.

## Sources

All code in `solution.py` was written for this task. The numerical constants
(Abramowitz & Stegun 7.1.26 for the complementary error function and 26.2.23
for the inverse normal CDF) are standard published approximations, transcribed
here so that no SciPy import is needed.
