# Long-Run Covariance Estimator

## Running

```sh
./run.sh /absolute/path/input.npy /absolute/path/output.npy
```

No build step or downloads are needed. The dependencies are Python 3, NumPy,
and SciPy, already installed in the supplied environment. Development used
Python 3.11, NumPy 1.26.4, and SciPy 1.12.0. `run.sh` fixes numerical-library
thread counts to one and uses the process's inherited CPU affinity.

Optional offline mathematical self-tests:

```sh
python3 -B selftest.py
```

The output is the long-run covariance **H**, not H divided by the number of
observations. It is a C-order little-endian float64 NumPy array in the input
coordinate order.

## Method

All estimates are fitted from the current input alone. Centering and SVD
whitening handle translations, scaling, and mixed coordinates.

For continuous observations, the program compares centered vector
autoregressions using a small-sample information criterion. It averages
competitive orders and evaluates each fitted spectrum at frequency zero.
One-lag fits are adaptively regularized toward a common autoregressive
coefficient when the data support pooling. The covariance is further
regularized using a delta-method uncertainty estimate, or a contiguous-block
jackknife for non-Gaussian innovations. A conservative flat-top check of the
prewhitened influence series can correct remaining low-frequency dependence.

For approximately Gaussian short-memory data, additional moving-average
likelihood fits are compared with the autoregression. Their stationary
finite-sample likelihood integrates out the initial disturbance and profiles
the unknown mean, avoiding frequency-domain boundary leakage near a spectral
zero. Schur-form filters keep this calculation in the installed SciPy
toolchain. Optional likelihood refinement has its own wall-clock budget.

If a trajectory repeatedly returns to a modest set of exactly recurring
states, the program instead estimates its transition probabilities and solves
the finite-state Poisson equation. Detailed balance is used only after a
likelihood test with a sparse-count correction. Block deletion and a precision
regularizer limit unstable estimates of very slow modes.
Transition counts use the observed endpoints; a closing-edge regularizer is
needed only when the empirical directed stationary law is degenerate.

All matrix regularization is performed in normalized coordinates. The final
estimate is projected onto the positive-semidefinite cone, symmetrized, and
mapped back to the original coordinates. A positive-semidefinite overlapping
batch estimate is available as a numerical fallback.

## Sources and Reproducibility

`estimator.py` and `moving_average.py` are original implementation code. NumPy
and SciPy supply the installed linear algebra, optimization, FIR filtering,
and chi-square quantile routines. No external datasets, reference covariances,
network services, saved fits, or additional chain simulation are used by the
program.
The numerical methods use no random numbers. Each invocation is independent
and writes only the requested output file.
