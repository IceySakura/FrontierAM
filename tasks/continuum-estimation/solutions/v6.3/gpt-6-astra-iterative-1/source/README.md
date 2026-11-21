# Continuum Estimator

This program estimates the zero-level limit from four or five positive,
possibly nonuniform discretization levels. It uses only the current input.

The primary model is

    y(x) = limit + a*x^p + b*x^(p+d) + c*x^(p+d+1.5),

with positive, unknown orders. Rather than selecting an arbitrary exact fit,
the estimator integrates over orders and regularizes the correction
coefficients. It includes the interpolation Jacobian in the integration
weights and returns a weighted median of the resulting limits.
Single-power and constant trends are recognized directly when all observations
verify the simpler model to numerical precision.

Four-point data use adaptive two-dimensional quadrature. Five-point data
determine curves of exact interpolants; two overlapping coordinate charts
integrate these curves without singular weights at turning points.
Calculations use normalized levels and responses. Coefficient and order
priors have flat central regions and soft tails, with a mixture over correction
signs. A low-weight broad-order component keeps more extreme orders available.
These are regularization assumptions informed by the public examples, not
restrictions on the input schema. Implausible fits fall back to a bounded local
power extrapolation. Nearly redundant levels are collapsed, and very tightly
clustered grids use local derivatives. Oscillatory and numerically degenerate
inputs have finite-output fallbacks.
An internal integration deadline leaves ample time for output within the
per-process limit.

## Running

The only dependencies are the supplied Python 3.11, NumPy, and SciPy runtime.
There is no installation, compilation, network use, or persistent state.

    /bin/sh /workspace/submission/run.sh

By default this reads `/workspace/inputs/input.json` and writes exactly one
numeric `estimate` field to `/workspace/result/output.json`. Optional input
and output paths are accepted:

    PATH=/workspace/runtime/bin:$PATH sh /workspace/submission/run.sh INPUT OUTPUT

The runner sets numerical libraries to one thread. All source files are
included in this directory.

## Checks

    /workspace/runtime/bin/python3 /workspace/submission/test_estimate.py

The tests use synthetic formulas, not stored answers or case recognition.
