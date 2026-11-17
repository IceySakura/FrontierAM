# Continuum estimation from positive discretization levels

`run.sh` reads `/workspace/inputs/input.json` (or the optional first argument),
writes `{"estimate": ...}` to `/workspace/result/output.json` (or the optional
second argument) and always exits successfully.

## Method

The public development truths show that every smooth case is reproduced to
machine precision by the three-term expansion

    y(x) = t + a x^p + b x^q + c x^(q + 1.5),   0 < p < q,

so the unknowns are the limit `t`, the coefficients `(a, b, c)` and the orders
`(p, q)`. With four or five levels the model is underdetermined (six unknowns),
so the program computes a Bayesian posterior for `t` and reports its
loss-weighted median (weights `1/max(1,|t|)`, matching the scoring loss).

* For fixed `(p, q)` the model is linear in `(t, a, b, c)`. All determinants
  needed on a dense `(p, q)` grid are obtained from cofactor vectors, so a whole
  grid is evaluated with a few matrix products (`est.py`).
* Four levels: every grid point has one exact solution; its posterior weight is
  `prior(p, q) * prior(t, a, b, c) / |det A(p, q)|` (exact-data limit of a
  Gaussian likelihood).
* Five levels: exact solutions lie on the curve `det[1, x^p, x^q, x^(q+1.5), y] = 0`.
  Grid points are weighted by `prior / |grad D|` times a narrow kernel in
  their distance `|D| / |grad D|` to the curve, which approximates the line
  integral of the exact-data posterior; coefficients come from least squares.
* Priors: empirical-Bayes "uniform with unknown bounds" predictive densities,
  flat over the ranges seen in the public examples with polynomial tails, plus
  sign constraints `b <= 0`, `c <= 0` observed in every smooth example.
* Fallbacks: if the main prior leaves no posterior mass, the posterior is
  recomputed under a very broad prior (no sign constraints). If that also
  fails, or anything raises, a generalized Aitken/Richardson estimate on the
  three finest levels is written, and as a last resort the value at the
  finest level. Output is always finite.

Only Python 3.11 + NumPy from the bundled runtime are used. The grid is
processed in p-chunks (peak memory ~100 MB); a case takes about 1-1.5 s.
`run.sh` pins BLAS/OpenMP to one thread and, as a last resort, writes a
finite placeholder if the program somehow produced no output.

## Development results

* Public smooth examples: mean loss ~1.4e-4 (generalized Richardson: 3.7e-3).
* Synthetic cases drawn from ranges consistent with the public examples:
  mean loss ~8.5e-4 vs ~1.1e-2 for generalized Richardson. Error is dominated
  by four-level cases with low leading order p, where the data cannot
  separate the terms and the prior decides.
