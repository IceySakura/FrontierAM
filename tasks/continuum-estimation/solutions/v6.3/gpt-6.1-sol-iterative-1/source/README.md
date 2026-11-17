# Continuum Estimation

This submission estimates the zero-level limit from four or five positive
discretization levels. It is entirely offline and uses the bundled Python
runtime with NumPy and, for the fallback fit, SciPy. There is no build step,
download, training table, persistent state, or additional simulator query.

## Running

The evaluator runs:

```sh
/bin/sh /workspace/submission/run.sh
```

This reads `/workspace/inputs/input.json` and writes
`/workspace/result/output.json`. Both paths can be overridden for development:

```sh
PATH=/workspace/runtime/bin:$PATH /bin/sh /workspace/submission/run.sh INPUT OUTPUT
```

The output contains exactly one finite numeric field, `estimate`. The launcher
sets numerical-library thread counts to one.

## Method

Observations are sorted by level. The primary estimator integrates a family
of smooth convergent error models:

```text
y(x) = limit + a*x^p + b*x^(p+d) + c*x^(p+d+1.5).
```

Compact order and amplitude priors regularize the underdetermined fit. All
observations contribute to each fit. Divided differences in log level remove
the unknown intercept, and `expm1` stabilizes differences between close levels.
For four observations, the integration coordinates are the order separation
and the higher-order amplitude. For five observations, the coordinates reduce
to the order separation. Batched, bracketed root finding resolves the remaining
order, and change-of-variable Jacobians supply the integration weights. A
loss-weighted posterior median supplies the point estimate.

Exactly supported single-power data are recognized with an all-observation
consistency check before applying a regularized mixture. If the compact family
cannot support the observations, a wider power family and then guarded
Richardson-style fits are tried. Constant data and clearly oscillatory data have
conservative fallbacks. Numerical failures cannot propagate a nonfinite value
into the output.
