# Long-Run Covariance Estimator

## Running

```
./run.sh /absolute/path/input.npy /absolute/path/output.npy
```

The input is one C-order little-endian float64 array of shape `(M, d)`.
The output is one C-order little-endian float64 array of shape `(d, d)`.
It is the long-run covariance `H`, not `H/M`.

## Dependencies

Python 3, NumPy, and SciPy, already installed in the offline environment.
No build step or `build.sh` is needed. No network access, package installation,
external data, additional trajectories, or persistent state is used.
`run.sh` restricts numerical libraries to one thread and disables bytecode
cache writes.

## Method

1. Center and scale the observations, then use an SVD to whiten the active
   coordinate space. This makes the estimation insensitive to coordinate
   scales and linear mixing.
2. For recurring finite-support observations, fit a regularized transition
   matrix and solve the finite-chain Poisson equation. Consecutive repetitions
   alone do not qualify a continuous Metropolis trajectory as finite-state.
3. Otherwise, fit conditional vector autoregressions up to order 32 using
   sufficient-statistic matrices. Average their zero-frequency spectral
   estimates with finite-sample information-criterion weights.
4. When a blocked diagnostic is consistent with reversibility, compare the
   unrestricted estimate with common-dynamics and grouped-dynamics fits.
   Pooling is controlled by mode uncertainty and penalized likelihood.
5. Regularize noisy long time scales and transform the symmetric positive
   semidefinite estimate back to the original coordinates.

All model fits and decisions use only the current input. Calls are
deterministic and independent. The implementation is original code using
standard NumPy and SciPy linear algebra routines; no third-party source or
reference matrices are bundled.

## Source Files

- `estimator.py`: input/output, normalization, finite-chain and VAR estimators.
- `regularize.py`: reversibility diagnostic and covariance-shape shrinkage.
- `shared.py`: likelihood-weighted common-dynamics estimator.
- `grouped.py`: conservative pooling of groups of similar dynamic modes.
