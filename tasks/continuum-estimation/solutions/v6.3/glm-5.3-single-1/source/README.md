# Continuum estimation from positive discretization levels

Estimates the limiting value `t` of a deterministic simulator `y(x)` as the
positive discretization level `x -> 0`, given four or five (possibly
non-uniform, unsorted) observations.

## Usage

```
/bin/sh /workspace/submission/run.sh                 # reads  /workspace/inputs/input.json
                                                     # writes /workspace/result/output.json
/bin/sh run.sh INPUT OUTPUT                          # optional explicit paths (development)
```

The output file contains exactly one field, e.g. `{"estimate": 2.2929041492}`.
The result directory is created if missing. Runtime is a few seconds per case
(well inside the 60 s budget), single-threaded, no network, no build step.

## Structure of the problem

Fitting the public development pairs revealed that every *smooth* case
satisfies, to machine precision (residual ~1e-16 against values of order 1-3),

```
y(x) = t + A*x**p + B*x**q + C*x**(q + 1.5),      0 < p < q
```

i.e. a three-term power expansion whose third exponent is exactly the second
exponent plus 1.5. This was verified by taking `p`, `q` and `t` from the truth
files, fitting only `A`, `B`, `C`: the residual is at rounding level for all
seven smooth dev cases, and it degrades quadratically as the 1.5 offset is
perturbed, which pins the offset. There is no observation noise.

Consequently the inference problem is an *under-determined exact nonlinear
system*: six unknowns (`t, A, B, C, p, q`) but only four or five equations. A
pure least-squares fit is therefore meaningless (with five points an exact fit
exists for a one-parameter family; with four points, for a two-parameter
family), and the answer must come from regularising with prior knowledge.

## Method

A Bayesian linear/Gaussian model is marginalised over a fine `(p, q)` grid:

* conditionally on `(p, q)`, the coefficients `theta = (t, A, B, C)` get a
  Gaussian prior with covariance `S = diag(sT^2, sA^2, sB^2, sC^2)` (`sT` large,
  i.e. essentially flat in `t`; `sA, sB, sC` weakly informative, matching the
  coefficient magnitudes seen in the development set);
* a small Gaussian noise level `sigma` (numerically tiny, but finite so the
  razor-sharp exactness of the model is expressed as an overwhelmingly sharp
  likelihood rather than a delta function);
* the exact marginal likelihood (evidence) and posterior mean are closed form
  through the 4x4 system `K = M^T M + sigma^2 S^-1`, `c = K^-1 M^T y`,
  `M = [1, x^p, x^q, x^(q+1.5)]`:

  `log ev(p,q) = -1/2 [ log det K + ||y - M c||^2 / sigma^2 ]`;

* the posterior of `t` is the mixture of the per-grid-point posteriors and the
  reported estimate is a blend of its mean and median (the mean is better for
  narrow posteriors, the median for the skewed posteriors produced by the
  four-point cases).

Everything is vectorised: the grid is processed in chunks of `(p, q)` values
with batched 4x4 solves and `slogdet`, so a 900 x 2800 grid costs ~2 s.

Grid/prior settings (`p in [0.15, 3.8]`, `q in [0.2, 6.0]` with `q > p`,
`sA = 1.1`, `sB = 0.65`, `sC` around 0.1-0.15, `sigma = 1e-6 * scale`) were
selected on the public development set and cross-checked on a synthetic
generator calibrated to the development statistics.

## Robustness

* The whole computation is guarded; any failure falls back to a classical
  generalised Richardson extrapolation (fit `y = t + c*x^p` on the smallest
  levels with `p` chosen by a one-dimensional search) and, in the last resort,
  to the smallest-level observation.
* A model-adequacy check monitors the best residual reached on the grid; if the
  input does not admit a near-exact fit of the identified family (for example
  the oscillatory audit case in the public set), the fallback estimator is used
  instead of the structured one.
* Inputs are validated (equal-length finite `x`/`y`, positive distinct `x`);
  the arrays are sorted internally, so input order is irrelevant.

## Files

* `run.sh` — entry point, selects the bundled numerical interpreter.
* `estimate.py` — the whole program (NumPy only; SciPy is used opportunistically
  in the fallback path).
