# Continuum estimation from positive discretization levels

Estimates the limiting value `t = lim_{x->0} y(x)` from 4 or 5 noise-free
observations at positive, distinct, possibly unsorted levels `x`.

## Method
The observations follow a smooth three-term convergence law

```
y(x) = t + a*x^p + b*x^q + c*x^(q + 1.5)
```

with unknown positive orders `p < q`. This structure was identified from the
public development pairs (the third exponent is exactly `q + 1.5`, and the fit
reproduces every dev observation to machine precision).

With only 4-5 points the six parameters are under-determined, so a single
nonlinear fit is unstable. Instead the program:

1. Sorts the observations by `x`.
2. Characterizes the exact-fit manifold:
   - `n = 4`: for a grid of `(p, q)` it solves the 4x4 linear system for
     `(t, a, b, c)` (every `(p, q)` fits exactly).
   - `n = 5`: it traces the zero-residual curve in `(p, q)` (the residual is
     one-dimensional, found via the left null vector of the basis matrix) and
     refines each branch.
3. Weights each feasible solution by a weak prior on the orders and
   coefficients (including the coarea/Jacobian factor `1/|det A|` for `n = 4`).
4. Reports the interpolated weighted median of `t`, which minimizes expected
   absolute error, the scored loss.

A guarded fallback (order-2 extrapolation) and a sanity clamp guarantee a
finite estimate for every input.

## Running
```
/bin/sh /workspace/submission/run.sh
```
Reads `/workspace/inputs/input.json` (`{"x":[...],"y":[...]}`) and writes
`/workspace/result/output.json` (`{"estimate": <finite number>}`). For local
development `run.sh INPUT OUTPUT` overrides the paths.

## Dependencies
Python 3.11 with NumPy 1.26.4 and SciPy 1.12.0 from the bundled offline runtime
(`python3` first on `PATH`). Single-threaded; no network, no build step. Each
invocation runs in well under a second of compute after interpreter startup.
