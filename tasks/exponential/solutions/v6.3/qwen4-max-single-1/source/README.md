# FAMEXP2 exponential propagation

This submission computes

```text
y = exp(-t L) b0 + t phi1(-t L) b1
```

for the streamed symmetric positive semidefinite CSR operator in the FAMEXP2
protocol.

## Algorithm

The solver uses several input-dependent paths:

* `t = 0` and zero operators are handled directly.
* Diagonal operators are evaluated elementwise with `exp` and `expm1`.
* Operators whose connected components have size at most two are evaluated
  exactly with closed-form 1x1/2x2 spectral formulas. Components up to size 64
  use small dense symmetric eigendecompositions.
* General sparse operators use a Chebyshev expansion on the Gershgorin interval
  `[0, max_i sum_j |L_ij|]`. The exponential and `t*phi1` coefficients are
  computed from scaled modified Bessel functions; the phi coefficients use a
  composite Gauss-Legendre quadrature for the Bessel integral. For the strictest
  precision level, the phi recurrence is evaluated in extended precision to
  suppress roundoff when the requested output is small.

The sparse recurrence is implemented in C++ and evaluates the two right-hand
side recurrences together, sharing each CSR matrix traversal. The degree is
chosen from a coefficient-tail bound and the requested tolerance.

## Files

* `solve.py`: protocol handling, coefficient generation, exact small-component
  paths, and dispatch.
* `fame_core.cpp`: C++ sparse Chebyshev recurrences and row-bound helper.
* `fame_core.so`: prebuilt shared library for the supplied Linux amd64 image.
* `build.sh`: offline rebuild of the shared library.
* `run.sh`: executable entry point. It rebuilds the shared library only if it is
  missing, then starts the solver.

## Dependencies

Only the supplied offline environment is used: Python 3, NumPy, SciPy, mpmath,
and g++. No network access or package installation is required.

## Local checks

From the task directory:

```sh
python tools/run_local.py --manifest samples/manifest.json \
  --command "/workspace/submission/run.sh" --repeats 3 \
  --output dev_report.json
```

The submission also exits immediately after the final requested vector and
writes diagnostics, if any, only to stderr.
