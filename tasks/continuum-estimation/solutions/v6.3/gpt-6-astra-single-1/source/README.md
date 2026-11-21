# Continuum Estimator

## Running

The evaluator entry point is:

```sh
/bin/sh /workspace/submission/run.sh
```

It reads `/workspace/inputs/input.json` and writes
`/workspace/result/output.json`. Optional positional arguments override the
input and output paths:

```sh
PATH=/workspace/runtime/bin:$PATH /bin/sh /workspace/submission/run.sh INPUT OUTPUT
```

The only numerical dependencies are the supplied NumPy and SciPy. There is no
installation, compilation, network access, learned-state file, or cross-run
state. The launcher limits numerical libraries to one thread and uses the
inherited CPU affinity.

## Method

Observations are sorted by positive discretization level. The main model is a
generalized power expansion,

```text
y(x) = limit + a*x^p + b*x^q + c*x^(q+1.5),   0 < p < q.
```

This expansion is motivated by the public smooth examples. With four or five
observations it is underdetermined, so the program integrates compatible fits
instead of selecting an arbitrary nonlinear least-squares solution. Bounded
order and coefficient priors favor moderate corrections, with a tapered tail
for very slow convergence. A symmetric component and a broader Gaussian
component allow other signs and coefficients.

For five observations, a null-space calculation and a bracketed scalar root
solve parameterize the interpolation curve by `q`; adaptive quadrature
resolves narrow regions of posterior mass. For four observations,
quadrature over `q` and `c` leaves a unique bracketed solve for `p`, followed by
linear coefficient recovery. Jacobian determinants give the appropriate
integration weights. The answer is a posterior weighted median, adjusted for
the specified normalized absolute-error loss.

Column rescaling, bounded root solves, chunked quadrature, and finite-value
checks control numerical and memory costs. Exact single-power trends have a
fast path. If the regular model is incompatible or numerically unreliable,
the program uses bounded single-power extrapolation or a conservative
combination of the finest observations. The output always contains just one
finite numeric `estimate`.

## Files

- `continuum.py`: complete numerical estimator and JSON interface.
- `run.sh`: offline, single-threaded launcher.
- `README.md`: this document.
