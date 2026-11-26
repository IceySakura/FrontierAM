# Continuum Estimator

This offline program estimates a zero-discretization limit from four or five
positive-level observations. It sorts the levels, fits a fractional-power
asymptotic expansion, and integrates over unidentified convergence orders and
coefficients. The reported estimate is a posterior median, with a bounded
local extrapolation fallback for incompatible data.

## Method

1. Sort the observations and recognize constant or exact single-power trends.
2. Use the expansion `y(x) = L + a*x**p + b*x**q + c*x**(q+1.5)`, with
   positive ordered exponents. Four or five observations do not identify all
   six parameters, so the program integrates over the remaining freedom.
3. Project out fixed basis functions using QR decomposition, solve the
   remaining scalar order equation by bisection, and recover the coefficients
   without forming normal equations.
4. Weight candidates by quadrature weights, the change-of-variables
   Jacobian, and soft coefficient/order priors. Both signs are allowed.
5. Return the posterior median weighted for the normalized absolute-error
   objective. Incompatible fits use a bounded local two-power fit. A final
   finite-value guard falls back to the finest observation.

The program reads only its current input at execution time. It contains no
case lookup tables, simulator queries, learned answer files, or persistent state.

## Run

```sh
/bin/sh /workspace/submission/run.sh
```

The default input is `/workspace/inputs/input.json`. The program creates
`/workspace/result/output.json` with exactly one finite numeric `estimate`.
Optional positional arguments replace the input and output paths:

```sh
PATH=/workspace/runtime/bin:$PATH sh /workspace/submission/run.sh INPUT OUTPUT
```

The implementation uses the installed Python, NumPy, and SciPy runtime.
There are no downloads, build steps, persistent state, or additional data files.
All numerical libraries are restricted to one thread.

## Verification

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /workspace/runtime/bin/python3 /workspace/submission/test_estimator.py -v
```

The tests cover exact powers, small positive orders, constants, permutations,
sign reversal, extreme numeric scales, and fresh-process JSON output on the
public examples. Representative local full-process runs take approximately
1-3 seconds and peak below 260 MiB. No CPU index is hard-coded.
