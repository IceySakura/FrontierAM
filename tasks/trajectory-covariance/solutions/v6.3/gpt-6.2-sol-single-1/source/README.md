# Long-run covariance estimator

The program reads one trajectory and writes its estimated long-run covariance
`H`. It does **not** divide by the number of rows. The output is a symmetric
positive semidefinite, C-contiguous, little-endian `float64` NumPy array.

## Build and run

Run from the submission directory:

```sh
./build.sh
./run.sh /absolute/path/input.npy /absolute/path/pred.npy
```

The build uses the installed C++17 compiler (`g++` by default) to compile
`innovations.cpp`, a small filter used by the vector ARMA fit. The runner uses
the installed Python 3, NumPy, and SciPy. No downloads, additional packages,
or agent tools are required. The numerical libraries are restricted to one
thread, and the program inherits its CPU affinity. The estimator is
deterministic.

## Method

- Recurrent finite states are modeled by transition counts, with separate
  regularization of holding rates and jump destinations. A stationary Poisson
  equation gives the long-run covariance of the observed state vectors. For
  nearly discrete observations, a clustering fit is used only if the states
  are well separated and the within-state residuals have little dependence.
- Continuous trajectories are centered and whitened for fitting, then mapped
  back to the input coordinates. Short-memory autoregressive and
  moving-average fits estimate the spectrum at zero. A regularized spectral
  correction can account for residual dependence.
- When supported by the observed transitions, reversible or nonlinear
  Poisson projections, conditional holding-time fits, and models for
  coordinate updates or random-walk Metropolis moves provide additional
  estimates. Empirical fit checks control when and how much these are used.

All implementation source is included and was written for this submission.
The program uses only the supplied trajectory. It does not read reference
matrices, generate additional trajectories, or retain data between calls.
