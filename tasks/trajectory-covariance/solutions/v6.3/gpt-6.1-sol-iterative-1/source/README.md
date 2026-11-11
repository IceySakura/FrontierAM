# Long-Run Covariance Estimator

## Running

```sh
./run.sh /absolute/path/input.npy /absolute/path/output.npy
```

No build step, network access, or package installation is required.
Dependencies are the installed Python 3, NumPy, and SciPy. Development
used Python 3.11, NumPy 1.26.4, and SciPy 1.12.0.

The output is the long-run covariance H, not H divided by trajectory
length. It is one C-order, little-endian float64 NPY array.

## Method

The estimator centers the trajectory and uses column scaling followed
by a data SVD to whiten its observed nondegenerate directions.
It combines independent-observation and first-order vector regression
spectral estimates using corrected Akaike weights. The vector regression
uses exact conditional least squares with an intercept, a sandwich
estimate of coefficient uncertainty, adaptive shrinkage toward a common
decay rate, and regularization of uncertain large spectral eigenvalues.

A likelihood-based common-rate fit is blended in when supported by a
homogeneity diagnostic. For a small, repeatedly visited finite state
space, an empirical transition model supplies a second estimate through
the Markov Poisson equation. Its circulation is shrunk toward
reversibility according to an empirical deviance, and its covariance
estimate is regularized and combined with the vector estimate.

Finally the matrix is transformed to the input coordinates and
symmetrized. The component estimates are positive semidefinite;
numerically negative spectral eigenvalues are removed. Constant input
returns zero. Numerical factorization failures are guarded.

## Execution And Sources

`run.sh` limits numerical libraries to one thread and sets `SEED=1`.
The implementation is deterministic, uses the inherited CPU affinity,
and does not read other trajectories or preserve information between
invocations. Working memory is linear in trajectory length, plus a
transition matrix with at most 512 states.

All estimator source is original to this submission. Its ingredients
are standard vector regression, Akaike weighting, covariance
regularization, finite-state Markov Poisson equations, and numerical
linear algebra supplied by NumPy and SciPy. There are no external
datasets, reference matrices, pretrained models, or vendored packages.
