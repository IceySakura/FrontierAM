# Long-run covariance estimator

## Run

From this directory:

```sh
./run.sh /absolute/path/input.npy /absolute/path/pred.npy
```

The input is an `M x d` NumPy array of ordered observations. The output is a
single `d x d`, C-order, little-endian float64 NumPy array with a symmetric
positive semidefinite estimate of the **long-run covariance** `H`, in the
original coordinate order. It is not divided by `M`.

This is original Python source that uses the installed Python 3, NumPy, and
SciPy. No build, downloads, external data, or network access are needed.
`run.sh` limits the numerical libraries to one thread. The estimator is
deterministic. It does not resample, share information between invocations,
or read any data other than the supplied trajectory.

## Method

- For observations that revisit a finite set of exact states, fit a transition
  operator from the visit and transition counts and solve its Poisson equation.
  The fit uses reversibility when supported by the transition counts and
  discounts directions that are poorly measured. Larger graphs use sparse
  linear algebra.
- For continuous observations, center and whiten the coordinates. Use a
  short-lag covariance estimate or a regularized vector autoregression
  in the linear branch. For approximately reversible transitions, project the
  Poisson equation onto linear and polynomial features. Discard poorly measured
  corrections using variation across trajectory blocks, and test for remaining
  dependence in the fitted Poisson residual.
- Pool similar relaxation rates when their spread is consistent with
  estimation noise. When both the acceptance rate and jump statistics support
  a Gaussian random-walk approximation, optionally fit a conditional shape
  from the observed moves and holds. Compute the Gaussian approximation's
  radial Poisson moments by deterministic quadrature.

## Sample check

```sh
./run.sh /workspace/samples/01/input.npy /tmp/pred.npy
python3 /workspace/samples/check.py /workspace/samples/01/reference.npy /tmp/pred.npy
```
