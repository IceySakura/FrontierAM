# Offline Conic Solver

Build and run:

```sh
./build.sh
./run.sh --input /absolute/problem.npz --output /absolute/result.npz
```

The implementation uses the preinstalled Python 3.12.14, NumPy 2.3.5,
SciPy 1.16.3, and the HiGHS 1.8.0 extension distributed inside SciPy.
The small native helper is compiled with the installed GCC 12.2 C++ compiler.
There are no network operations or downloaded dependencies.

`build.sh` compiles the native helper and prepares Python import caches.
All generated build artifacts remain in this directory. Solving is
single-process and numerical libraries are restricted to one thread.
The inherited CPU affinity is left unchanged.

## Methods

* Linear optimization uses HiGHS, with explicit alternative problems for
  infeasibility certificates.
* Suitable polyhedral QPs use an objective-scaled HiGHS active-set solve.
  Other quadratic and symmetric-cone problems use a homogeneous primal-dual
  predictor/corrector method, equilibration, reusable native sparse LDL
  factors, and iterative refinement. A small-QP operator-splitting fallback
  includes active-set polishing.
* Lorentz scalings use extended-precision geometry and sparse low-rank
  augmented systems to avoid cancellation close to the boundary.
* Semidefinite blocks use isometric upper-triangle coordinates,
  Nesterov--Todd scalings, and dense Schur complements formed by native code.
* Exponential cones have a normalized homogeneous barrier-Newton fallback.
  Recognized exponential epigraphs are reduced algebraically to convex
  log-sum-exp constraints, solved by a dense reduced primal-dual method,
  and reconstructed in the original conic coordinates.
* Sparse equality substitutions and constant-row removal are reversible.
  Very large models also have an exact null-cost feasibility presolve.
  Recognized weighted least-squares/L1 epigraphs use native coordinate
  descent with full primal and dual recovery.
  Termination checks use residuals, cone membership, and primal/dual gaps;
  infeasibility outputs use homogeneous certificates.

The public development examples and the supplied independent checker are
used for testing. This is a locally implemented solver, not a claim of
complete coverage of every ill-conditioned conic problem.

## References

* S. Mehrotra, "On the Implementation of a Primal-Dual Interior Point Method,"
  SIAM Journal on Optimization, 1992.
* Y. Nesterov and M. J. Todd, "Self-Scaled Barriers and Interior-Point Methods
  for Convex Programming," Mathematics of Operations Research, 1997.
* A. Skajaa and Y. Ye, "A Homogeneous Interior-Point Algorithm for
  Nonsymmetric Convex Conic Optimization," Mathematical Programming, 2015.
* P. Goulart and Y. Chen, "Clarabel: An Interior-Point Solver for Conic
  Programs with Quadratic Objectives," 2024.
* T. A. Davis, "Algorithm 849: A Concise Sparse Cholesky Factorization
  Package," ACM Transactions on Mathematical Software, 2005.
* B. Stellato et al., "OSQP: An Operator Splitting Solver for Quadratic
  Programs," Mathematical Programming Computation, 2020.
* The installed HiGHS implementation and SciPy's sparse and dense linear
  algebra interfaces.
