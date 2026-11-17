# Continuum Estimator

This is an offline, single-threaded Python program using the bundled NumPy and
SciPy runtime. No installation, network access, build step, or persistent state
is required. The launcher selects `/workspace/runtime/bin/python3` explicitly.

## Run

The grading entrypoint is:

```sh
/bin/sh /workspace/submission/run.sh
```

It reads `/workspace/inputs/input.json` and creates
`/workspace/result/output.json`, containing exactly one finite numeric
`estimate` field.

Optional paths are supported for local testing:

```sh
/bin/sh /workspace/submission/run.sh INPUT OUTPUT
```

## Method

The observations are sorted by their positive discretization levels. A
single-power variable-projection fit first handles exact power-law trends.
The main estimator uses a constant plus three powers with unknown leading
order and order separation. Its last power is 1.5 above the second, a smooth
correction structure supported by the public examples.

The linear coefficients are eliminated before searching over the orders.
With five observations, the remaining interpolation constraint is integrated
along its root branches. With four observations, integration over the last
coefficient avoids undersampling thin admissible regions. Jacobian weights
and bounded, mixed coefficient priors regularize the non-unique extrapolation.
A weighted median targets normalized absolute error.

Powers and power differences are evaluated in scaled logarithmic coordinates.
Computation is batched to bound memory use. Constant data and unsuccessful or
ill-conditioned model fits have finite fallbacks. The program neither reads
development truths nor recognizes particular cases.

Root counts are bounded and an internal work cutoff preserves time for output.
On the public examples, full launches take approximately 5-8 seconds.
