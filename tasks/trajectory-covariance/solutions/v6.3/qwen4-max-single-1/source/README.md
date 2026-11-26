# Long-run covariance estimator

`run.sh INPUT.npy OUTPUT.npy` reads one `(M, d)` trajectory and writes a
finite, symmetric, positive-semidefinite `(d, d)` long-run covariance matrix
`H`, so `H / M` estimates the covariance of the trajectory mean.

## Dependencies

Python 3, NumPy, and SciPy.  No network access or third-party downloads are
needed.  `build.sh` only checks that NumPy and SciPy import successfully.
The launch script forces single-threaded BLAS/FFT execution and uses the
inherited CPU affinity.

## Method

The estimator is generic and uses only the supplied trajectory:

* If the observations take a small number of distinct vector values, it
  estimates a finite-state transition matrix with a weak Dirichlet smoother
  and computes the stationary Markov-chain long-run covariance through the
  Poisson equation.
* Otherwise it estimates autocovariances by FFT.  A whitened, basis-invariant
  cutoff diagnostic identifies short moving-average behavior; a VAR(1)
  residual check prevents weak autoregressions from being mistaken for a
  cutoff.  Confirmed short moving averages use the corresponding truncated
  autocovariance sum.
* Remaining continuous trajectories use a vector autoregression.  Orders
  through 30 are fitted by multivariate Yule-Walker equations, scored with a
  small-sample AIC correction, and averaged with Akaike weights.  The
  zero-frequency VAR spectrum gives the long-run covariance.
* If VAR residuals are strongly non-Gaussian, the estimate is blended with a
  basis-invariant quadratic-spectral estimator; Gaussian linear residuals leave
  the VAR estimate unchanged.
* The final matrix is symmetrized and projected onto the positive-semidefinite
  cone to remove numerical roundoff.

## Run

```sh
./run.sh /absolute/input/input.npy /absolute/output/pred.npy
```

The output is a C-order little-endian float64 `.npy` file without pickle.
