# Continuum estimation from 4-5 positive discretization levels

`run.sh` reads `/workspace/inputs/input.json` (`{"x": [...], "y": [...]}`) and writes
`/workspace/result/output.json` (`{"estimate": <finite number>}`). For development:
`sh run.sh INPUT OUTPUT`. Uses the bundled Python 3.11 + NumPy only; about 1 s per case.

Files: `run.sh` (entry point), `continuum.py` (I/O, fallbacks, watchdog),
`estimator.py` (the method).

## Model

The public examples are reproduced to machine precision by the error expansion

    y(x) = t + a x^p + b x^q + c x^(q+1.5),   0 < p < q,

with unknown orders `p`, `q = p + d` and unknown coefficients; `t` is the continuum limit.
With 4-5 exact observations the model is under-determined (6 unknowns), so `t` is
estimated by Bayesian marginalization rather than by a single fit.

## Method

* For fixed `(p, d)` the model is linear in `beta = (t, a, b, c)`.
* **4 levels:** `beta = X(p,d)^{-1} y`; the exact-data marginal likelihood is
  `pi(beta) / |det X|`. The posterior over `(p, d)` is evaluated on a 2-D grid
  (step 0.005).
* **5 levels:** exact data confine `(p, d)` to the curve `det[X(p,d), y] = 0`, and the
  posterior is the line integral of `pi(p,d) pi(beta) / |grad det[X, y]|` along it,
  computed with marching squares on the same grid; `beta` is the least-squares
  solution at each segment midpoint.
* Levels are rescaled by `max(x)` for conditioning; the Jacobian factor
  `max(x)^-(3p+2d+1.5)` restores the density of the unscaled coefficients.
* Priors: smoothed uniform boxes `p in [0.5, 2.9]`, `d in [0.4, 2.4]`, `a in [-1.6, 1.6]`,
  `b in [-0.95, 0]`, `c in [-0.42, 0]`, flat in `t` (chosen from the public examples
  and checked on synthetic data drawn from a range of plausible generators).
* Point estimate: posterior median of `t` weighted by `1/max(1,|t|)`, which minimizes
  the expected task loss `|z-t|/max(1,|t|)`.

A self-consistency check (synthetic data drawn from the prior) gives uniformly
distributed posterior quantiles of the true `t`, `p` and `d`, which validates the
Jacobians and the curve integration.

## Robustness

A valid output is written immediately (value at the finest level), then replaced by a
3-level Richardson estimate, then by the posterior estimate. If the posterior is empty or
implausible (farther than `max(10*range(y), 3)` from the finest-level value), the priors
are widened and retried; a 40 s alarm guards the time limit. Inputs need not be sorted;
inputs with more than 5 levels use the 5 finest, fewer than 4 use the fallbacks.
