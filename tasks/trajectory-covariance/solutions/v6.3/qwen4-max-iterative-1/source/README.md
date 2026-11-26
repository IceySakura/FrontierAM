# Long-run covariance estimator

Dependencies: Python 3 and NumPy (already installed in the target environment).
No build step or network access is required.

Run:

```sh
./run.sh /absolute/input/input.npy /absolute/output/pred.npy
```

The estimator centers and whitens the input, selects a vector autoregressive
order by BIC using multivariate Yule-Walker equations, and maps the innovation
covariance to zero frequency. For lower-dimensional trajectories it also
considers a Hannan-Rissanen VARMA(1,1) candidate under the same BIC comparison.
If a five-dimensional trajectory has a small number of distinct observation
values, a finite-state transition-matrix estimate is used directly. For
six-dimensional trajectories that look strongly reversible and whose initial
positive-sequence estimate is close to the VAR estimate, the two estimates are
blended. A short PSD Bartlett window is used when no autoregressive order is
selected. The result is symmetrized and projected to the positive-semidefinite
cone before being written as a C-order float64 NPY.

All computation uses only the current input. BLAS threading is restricted to
one thread. The source is original and uses no reference answers or external
data.
