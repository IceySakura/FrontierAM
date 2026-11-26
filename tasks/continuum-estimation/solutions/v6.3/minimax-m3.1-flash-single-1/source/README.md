# Continuum estimation from positive discretisation levels

Estimates the limit `t = lim_{x->0+} y(x)` from observations `(x_i, y_i)` at
four or five distinct positive discretisation levels.

## Contents

| file | purpose |
|------|---------|
| `run.sh` | entry point; `run.sh [INPUT [OUTPUT]]`, defaults to `/workspace/inputs/input.json` and `/workspace/result/output.json` |
| `estimate.py` | the estimator (single file, NumPy only) |

`run.sh` prefers the bundled runtime `/workspace/runtime/bin/python3`
(Python 3.11.15 + NumPy 1.26.4) and otherwise falls back to the interpreter
on `PATH`.  It creates the output directory and always writes exactly one
finite number to the output file.

## Method

### 1. The error expansion

The task supplies only positive levels, so the whole problem is the shape of
the error `e(x) = y(x) - t` as `x -> 0`.  Fitting candidate expansions to the
eight public development pairs (`/workspace/public/dev`) shows that the
simulator follows

```
y(x) = t + a*x^p + b*x^q + c*x^(q + 3/2)
```

with two unknown positive convergence orders `p < q` (values in the public
truth files) and three amplitudes.  The evidence for that structure:

* `det[1, x^p, x^q, x^(q+3/2), y]` vanishes to `1e-30` at the published
  `p` and `q`, i.e. the five-parameter linear system reproduces the five
  observations to machine precision;
* the third exponent was recovered independently: scanning a free third
  exponent for the best four-parameter fit returns `q + 1.5` to six digits in
  every five-point development case;
* no two-term expansion (`t + a x^p + b x^q`) fits: its residual is
  `1e-4 ... 7e-4`, two to three orders of magnitude above machine
  precision, and adding `x^(q+3/2)` as a fourth column drops it to `1e-15`.

The amplitudes of the development cases are

```
a in [-1.43,  1.49]     b in [-0.87, -0.12]     c in [-0.37, -0.05]
```

which is consistent with `a ~ U(-1.5, 1.5)`, `b ~ U(-1, 0)`,
`c ~ U(-0.5, 0)`; those supports are used as the prior below.  The orders
behave like `p ~ U(0.5, 3)` and `q ~ U(1, 4.5)` with `q > p`.

### 2. The data do not identify a single model

The model has six unknowns but only four or five observations, so a family
of models reproduces the data exactly:

* **five levels** - the admissible `(p, q)` form a one-dimensional curve.  It
  is found with the determinant above, which changes sign across the curve,
  so sign changes on a scan grid bracket the curve and a bracketed root
  finder refines each crossing;
* **four levels** - the design matrix `[1, x^p, x^q, x^(q+3/2)]` is square,
  so *every* `(p, q)` reproduces the four observations exactly.

### 3. Selecting among admissible models

Admissible models are weighted by

* the **prior** on the orders and amplitudes (hard support, as estimated
  above);
* the **volume factor** `1/det` (four levels) or `1/pseudo-determinant`
  (five levels) of the design matrix.  This is the density of the posterior
  over the order plane: near-degenerate designs leave more freedom in the
  amplitudes and therefore carry more posterior mass.

The reported limit is the **weighted median** of `t` over the admissible
models.  The median (rather than the mean) is the point estimate that
minimises the expected absolute error, which is the scored loss.

If the assumed support excludes every admissible model - which would mean
the real amplitudes lie outside it - the prior is widened by factors
`1.25, 1.6, 2, 3, 8` until models become admissible, so the program never
fails.  A plain median over all candidates and a Richardson-style
extrapolation are kept as final fallbacks, and the output is always a single
finite number.

### 4. Accuracy

Measured on the eight public development pairs and on synthetic cases drawn
from the recovered generator, the relative error `|z - t| / max(1, |t|)` is

| data | mean | median |
|------|------|--------|
| public dev pairs (smooth family) | `1.1e-4` | `4.0e-5` |
| synthetic (same generator) | `6.9e-4` | `1.9e-5` |

For reference, on the same development pairs a fixed-order Richardson
extrapolation gives `2.8e-2` and a non-linear two-term fit `4.2e-2`
(average relative error).

Runtime is about one second per case (single threaded, NumPy only), well
inside the 60 second limit.
