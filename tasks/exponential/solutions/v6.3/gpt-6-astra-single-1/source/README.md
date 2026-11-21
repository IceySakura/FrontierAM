# Sparse Exponential Propagation

This is a standalone C++17 implementation of the FAMEXP2 streaming protocol.
It computes `exp(-t L) b0 + t phi1(-t L) b1` for symmetric positive
semidefinite CSR matrices, including singular matrices and signed entries.

## Build And Run

Run `bash build.sh` once, then `./run.sh`. The evaluator may run the supplied
build script before starting sessions. A compiled Linux amd64 executable is
also included.

Production dependencies are GCC's C++17 toolchain, the C++ standard library,
libm, and POSIX input/output. No Python, BLAS, downloads, agent tools, or
external data files are needed by the solver. The program inherits CPU
affinity and uses one thread.

## Algorithm

Diagonal and disjoint two-by-two blocks use extended-precision direct
formulas. Approximate eigenvector inputs have a residual-controlled scalar
path. Neither path assumes an invertible matrix or a graph Laplacian.

The general method is a combined Chebyshev expansion with one sparse
multiply per degree. It uses a fused Reinsch/Clenshaw difference recurrence,
an absolute row-sum spectral bound optionally tightened by positive diagonal
similarities, and extended-precision coefficient construction. Modified
Bessel coefficients come from a backward ratio recurrence. The phi1
coefficients are obtained by backward double summation. Degree selection
uses the nonnegative coefficient tails and input two-norms; the resulting
bound is checked against the requested output infinity-norm tolerance.
Cancellation-sensitive requests use an extended-precision recurrence.

Selected stiff systems first try a short-recurrence Lanczos projection.
Stopping bounds use shifted tridiagonal resolvents and semigroup
contractivity. Small projected matrix functions are evaluated in extended
precision with the same polynomial coefficients. Slow convergence falls
back to the general method, and unsuccessful trials are not repeated on
later requests in that session.

All matrix-dependent preparation occurs after READY. No state survives
between sessions. Standard output contains only protocol bytes; optional
diagnostics can be enabled with `PROP_DIAGNOSTICS=1`.

## Checks

`python selftest.py` runs independent small protocol and numerical checks
using the installed NumPy and SciPy. It covers sequential requests, zero
time, zero and subnormal operators, singular diagonals, signed diffusion,
rank deficiency, and cancellation.

Development also used the public FAMEXP2 harness: all six provided cases
passed three fresh-process repetitions. Generated checks included matrices
with 110,000 rows and about 990,000 nonzeros, 120,000-row hub and rank-deficient
operators, and 20,000-row wide-band operators, at stiffness up to 100,000.
References used Fourier/cosine diagonalization, analytic formulas, and an
independent extended-precision forward Chebyshev recurrence. Full-size
p10 reuse checks took approximately 11 and 44 seconds locally; a
120,000-row cancellation check took approximately 4.5 seconds.
A full-size 16-request cancellation session passed in 70.2 seconds with
approximately 147 MiB peak memory, below the 150-second and 4 GiB limits.
These are development measurements, not hidden-set results or a score.

No test matrices or reference answers are embedded in the solver.
