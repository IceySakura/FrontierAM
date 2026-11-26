# Continuum Estimator

This deterministic, single-CPU program estimates the value at zero from four
or five positive, possibly nonuniform discretization levels. Input observations
are sorted together; no grid spacing or input ordering is assumed.

## Method

Constant and single-power trends are handled directly. Other trends first use
variable-projection power-law fits, with multistart optimization for five levels
and an order-gap constraint for four levels. Very small, poorly identified
four-level orders are regularized.

Five-level inputs compare the two-power fit with a three-power error expansion.
The richer model integrates over positive convergence orders and regularized
coefficients. At fixed second order, a bracketed leading-order solve and batched
QR factorization reduce the calculation to one-dimensional quadrature.
Model evidence combines the candidate limits, and a weighted median minimizes
posterior normalized absolute loss. A substantially mismatched four-level fit
uses analogous order-gap marginalization.

Unsupported or ill-conditioned fits fall back to simpler extrapolation or the
finest observation. Valid inputs produce a bounded finite number. The program uses
only the current input and its source files, with no persistent state, simulator
queries, case lookup, network access, or downloaded dependencies.

## Running

The bundled Python 3.11.15, NumPy 1.26.4, and SciPy 1.12.0 runtime is sufficient.
No installation or build step is needed.

```sh
PATH=/workspace/runtime/bin:$PATH /bin/sh /workspace/submission/run.sh
```

The default input is `/workspace/inputs/input.json`. The program creates
`/workspace/result/output.json`, containing exactly one finite numeric
`estimate` field.

The input object contains `x` and `y` numeric arrays of equal length, either
four or five. Levels in `x` must be positive and distinct; all values must be
finite.

For local checks, optional paths are accepted:

```sh
PATH=/workspace/runtime/bin:$PATH /bin/sh /workspace/submission/run.sh INPUT OUTPUT
```

All numerical libraries are restricted to one thread by the launcher.
