# Long-Run Covariance Estimator

## Run

```sh
./run.sh /absolute/path/input.npy /absolute/path/output.npy
```

The output is the estimated long-run covariance **H**, not H divided by the
trajectory length. It is a single C-order, little-endian float64 NumPy array.

## Dependencies And Build

Python 3, NumPy, and SciPy are the only dependencies. The implementation was
developed with Python 3.11, NumPy 1.26.4, and SciPy 1.12.0, already installed in
the supplied environment. No compilation, downloads, network access, agent
tools, or optional packages are required. There is no build step.

The launcher limits numerical libraries to one thread and preserves the
process's inherited CPU affinity. All calculations are deterministic.

## Method

The trajectory is centered and standardized in its empirical covariance
geometry. The estimator then uses models inferred solely from that input:

* Repeated visits to a small finite alphabet permit estimation of a transition
  matrix. Holding probabilities are regularized with an empirical-Bayes
  beta-binomial fit. A discrete Poisson equation gives the covariance, evaluated
  through a positive-semidefinite martingale residual formula.
* Continuous trajectories are fitted with an automatically selected vector
  autoregression. A block jackknife estimates uncertainty and regularizes
  unstable covariance directions.
* A positive-semidefinite spectral mixture is also fitted. A penalized
  likelihood comparison selects it only when it improves on the autoregression.
  This addresses weak, slow components mixed with fast noise.
* For the autoregressive branch, a regularized low-frequency residual test can
  add covariance left unexplained by the fitted dynamics.

Final symmetrization and eigenvalue safeguards enforce the output contract.
Rank-deficient inputs and numerical fitting failures have finite
positive-semidefinite fallbacks.

## Source

`estimator.py` and `run.sh` are original source written for this submission.
They use NumPy and SciPy numerical routines; no third-party source is vendored.
No sample reference matrices, trajectory identifiers, model-specific answers,
or information from previous invocations are used.
