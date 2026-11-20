# Offline conic solver

Build and run from this directory:

```sh
./build.sh
./run.sh --input /path/to/instance.npz --output /path/to/result.npz
```

`run.sh` launches one numerical process with one BLAS and HiGHS thread. It
uses the inherited CPU affinity. It reads the version-1 NPZ input and writes
a version-1 NPZ with either a solution, a certificate, or `unresolved`.
It does not require any writes to the submission directory at run time.

## Offline dependencies

- Debian 12 on Linux amd64, GNU C++ 12.2, GNU binutils, zlib, and Jansson.
- Python 3.12.14, NumPy 2.3.5 and SciPy 1.16.3 as installed in the image.
- The OpenBLAS/LAPACK, SuperLU, and HiGHS 1.8.0 components bundled with that
  SciPy installation. The build enables a small native HiGHS bridge only
  for the exact fingerprinted wheel. It falls back to SciPy's Python
  bindings when the optional native bridge is unavailable.

The build does not download or install anything.

## Solver

The primary conic solver is a primal-dual predictor-corrector on a homogeneous
embedding that retains the quadratic objective. It uses Nesterov--Todd
scaling on symmetric cones and a logarithmically homogeneous dual barrier
with secant updates for exponential cones. The sparse Newton solver reuses
symbolic LDL structure and uses pivoted dense trailing blocks and refinement.
Wide Lorentz cones use rank-two updates; smaller ones use an orthogonal
basis that separates the largest and smallest cone-scaling eigenvalues.

Presolve eliminates fixed variables, contracts suitable pairs of equalities,
and reduces selected cone slices to linear or quadratic problems. Sparse PSD
blocks can split into chordal cliques, with dual matrix completion at recovery.
Dense PSD contributions use a Schur complement or a QR solve on the
transformed constraint basis. Separable bounds and certain diagonal and
bound-constrained QPs have direct or projected routines.

HiGHS simplex and IPM handle polyhedral problems alongside the native QP
path. Recovery options include cone-face Newton steps, projected splitting,
matrix-free linear solves, a pivoted sparse factorization, and valid
supporting-plane relaxations. Every accepted result, including a lifted
presolve result, is checked in the original input coordinates.

## Offline checks

The supplied independent checker accepted all five available sample cases
and all six positive interface examples. Generated tests exercised
infeasibility certificates in each cone family, moderate and extreme
equilibration, sparse PSD completion, and a one-million-variable diagonal
QP. An equivalent-rescaling test of the five samples verified 19 of 20
variants. Local timings are diagnostic, not a claim about the held-out score.

## Algorithm references

- Goulart and Chen, *Clarabel: An interior-point solver for conic programs
  with quadratic objectives*.
- Nesterov and Todd, self-scaled barriers and primal-dual scaling; Mehrotra,
  predictor-corrector interior-point iterations; Davis, sparse LDL.
- Dahl and Andersen, primal-dual interior-point methods for the exponential
  cone; Fukuda et al., sparse semidefinite programming by matrix completion.
- O'Donoghue et al., operator splitting with homogeneous self-dual embedding;
  Stellato et al., *OSQP: An operator splitting solver for quadratic programs*.
- SciPy's bundled HiGHS, SuperLU, and OpenBLAS/LAPACK implementations.
