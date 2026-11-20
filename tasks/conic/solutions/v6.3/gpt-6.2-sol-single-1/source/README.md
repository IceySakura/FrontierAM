# Offline conic solver

From the submission directory:

```sh
./build.sh
./run.sh --input /absolute/path/to/input.npz --output /absolute/path/to/output.npz
```

The executable reads schema-version 1 NPZ files and writes a solution, a
primal or dual infeasibility certificate, or `unresolved`. The timed entry
point is native C++ and runs numerical code on one thread, with inherited CPU
affinity. It does not write temporary files or use any state from prior calls.
Set `CONIC_DEBUG=1` for diagnostic output during development.

## Offline dependencies

The target is the supplied Debian 12 Linux amd64 image. Building uses GCC /
G++ 12.2, Python 3.12.14 (`sysconfig` and other standard-library modules), and
GNU binutils. The executable uses the image's zlib 1.2.13 and the installed
SciPy 1.16.3 wheel's OpenBLAS/LAPACK, HiGHS 1.8.0, and sparse ordering routines.
`dependencies.py` resolves the version-specific numerical entry points from
that wheel at build time. The native adapter is specific to the installed
wheel; the solver does not start a Python interpreter at run time. No
packages are downloaded or installed by the build.

## Algorithms

- General conic path: homogeneous primal-dual predictor-corrector steps,
  cone-preserving equilibration, Nesterov-Todd scaling for symmetric cones,
  and primal-dual exponential scaling with a dual-barrier alternative.
  The reduced Newton system uses reusable sparse LDL symbolic factors or
  pivoted dense factors, a sparse extension for Lorentz and exponential
  blocks, and Schur complements for larger PSD blocks.
- Reversible presolve handles fixed variables, single-link equalities,
  two-dimensional Lorentz blocks, certain constant cone faces, independent
  PSD components, and the dual orientation of PSD constraints with an
  explicit slack identity. Linear problems use HiGHS, including recovery from
  compact variable bounds.
- Fallbacks include operator splitting with matrix-free preconditioned
  conjugate gradients, active-manifold Newton polishing, and incremental
  supporting planes for exponential and small Lorentz blocks. Infeasibility
  recovery can use rank-revealing QR and normalized feasibility problems.
- Special forms have separate paths: diagonal-shift SDPs, sparse box QPs,
  projection-form cone problems, and single-epigraph log-sum-exp models.
- Every candidate is checked against the original matrix, cone blocks, and
  objective gap or certificate before it is returned.

Algorithm references: Mehrotra (1992), *On the Implementation of a
Primal-Dual Interior Point Method*; Nesterov and Todd (1997), *Self-Scaled
Barriers and Interior-Point Methods for Convex Programming*; Dahl and
Andersen (2021), *A Primal-Dual Interior-Point Algorithm for Nonsymmetric
Exponential-Cone Optimization*; Goulart and Chen (2024), *Clarabel: An
Interior-Point Solver for Conic Programs with Quadratic Objectives*; Davis
(2005), *Algorithm 849: A Concise Sparse Cholesky Factorization Package*;
Boyd et al. (2011), *Distributed Optimization and Statistical Learning via
the Alternating Direction Method of Multipliers*; and Wang, Chang, and Kolter
(2017), *The Mixing Method: Low-rank Coordinate Descent for Semidefinite
Programming with Diagonal Constraints*.

## Development checks

The workspace provided five development inputs and six valid interface
examples in `samples/`. Each passed three fresh runs with the bundled
independent checker and consistent statuses. Additional generated problems
exercised planted KKT points, coefficient scales, infeasibility directions,
SDP dualization, and large sparse problems, including a one-million-variable
box QP. These are local checks, not measurements on the held-out set.
