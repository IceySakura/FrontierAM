# Sparse Exponential Propagation

This is an offline C++17 implementation of the FAMEXP2 streaming interface.
Build with `./build.sh`, then execute `./run.sh`. The run script also builds a
missing executable before emitting READY. No Python packages, downloads, agent
tools, or network access are needed. Dependencies are the installed C++ compiler,
standard library, and Eigen headers at `/usr/include/eigen3`.
The offline build uses `-march=native` for the grading machine's CPU; the program
does not assume any particular affinity index.

The general solver expands the exponential and phi1 together in a Chebyshev
basis on a safe spectral interval. Scaled modified Bessel coefficients are
computed by a normalized backward recurrence in extended precision; phi1
coefficients follow from positive tail sums. A single stabilized Clenshaw
recurrence evaluates the sum of both matrix functions. Truncation is controlled
using the Euclidean norms of the actual right-hand sides and checked against
the requested infinity-norm tolerance. Stiff p10 evaluations use extended
precision when needed. Diagonally dominant matrices use a difference-form
multiply that preserves small row residuals and avoids cancellation at zero
eigenvalues. A positive-vector weighted Gershgorin bound tightens the interval
for stiff operators without making spectral assumptions.

Disconnected components of at most 32 vertices are diagonalized once in
extended precision and reused for every request. Zero matrices and zero time
are handled directly. Matrices may be singular and may have signed entries.
If a positive Gershgorin lower bound proves the transient negligible, a
Jacobi-preconditioned conjugate-gradient solve is accepted only after an
extended-precision true-residual check certifies its error.
An additional invariant-vector path uses scalar matrix functions only when an
ODE-defect bound certifies the complete output against the requested tolerance.
All input-dependent work occurs after READY. The program inherits CPU affinity
and does not create threads or retain state across sessions.

Public checks are run with the supplied `tools/run_local.py` harness. Additional
development checks compare small operators with dense eigendecomposition and
exercise large sparse operators, all tolerance levels, and sequential requests.

## Development Checks

- All six supplied samples passed three fresh-process repetitions.
- Dense symmetric PSD, signed, singular, rank-one, and ill-conditioned cases
  were cross-checked against extended-precision eigendecomposition.
- Analytic Fourier checks covered diffusion and signed/permuted diffusion,
  cancellation, forcing-only, zero forcing, and nonzero diagonal potentials.
- Zero time, zero matrices, subnormal data, and 16-request small-block sessions
  passed at the requested tolerances.
- A 120,000-row, 840,000-nonzero, non-diagonally-dominant p10 session at stiffness
  100,000 passed all 16 sequential requests in 84.5 seconds on the development
  host. These are development measurements, not a hidden-set score.
