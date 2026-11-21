# Cubature Submission

Run `./run.sh`. The program reads the initialization message and oracle replies
from standard input and emits only JSON Lines queries and a final result on
standard output. It needs the installed Python 3, NumPy, and SciPy; it does not
download packages or access test data.

The estimator uses queried values to select and validate low-complexity
surrogates: polynomial and exponential-polynomial models, separable factors,
sparse Fourier expansions, piecewise-affine ridge expansions, and positive
Gaussian mixtures. Analytic integrals are used when available. Unresolved
functions use an integrated kernel surrogate with an independent scrambled
Sobol residual estimate.

All oracle access goes through a single point-budget counter with batches of at
most 512 points. Model fitting, derivative probes, validation, and residual
samples share that budget. Numerical work has an internal time limit, with a
finite estimate from actual sampled values retained as a fallback. The launcher
limits numerical-library threading and preserves the inherited CPU affinity.
