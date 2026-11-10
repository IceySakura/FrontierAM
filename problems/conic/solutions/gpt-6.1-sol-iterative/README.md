# Portable Conic Solver

## Build and Run

```sh
./build.sh
./run.sh --input /absolute/input.npz --output /absolute/result.npz
```

The build compiles the C++ numerical/presolve helper and prepares reusable
Python bytecode. It does not inspect any problem input. The run script uses
`exec`; all problem-dependent work is performed in the solver process.
Input and output use the schema-1 NPZ interface, without pickle.

Dependencies are the preinstalled Python 3.12.14, NumPy 2.3.5, SciPy 1.16.3,
and g++ 12.2.0. SciPy supplies HiGHS 1.8.0, SuperLU, BLAS and LAPACK.
The target is Linux amd64. No network access or package installation is
required. Numerical libraries are restricted to one thread and CPU affinity
is inherited.
Startup avoids importing NumPy's unused testing and Fortran-compilation
front ends into SciPy's compatibility namespace; NumPy's exports are
restored immediately after that import.

## Methods

- Sparse affine union-find and singleton-equality substitution remove
  redundant lifted variables. Primal vectors and adjoint multipliers are
  restored to the original coordinate system.
  Constants introduced by substitution are retained in the relative-gap
  scale, separately from the input objective offset.
- Polyhedral problems use HiGHS simplex, IPM and convex QP routines, with
  original-system KKT refinement. Bounded alternative LPs recover Farkas
  separators or recession directions when needed.
- A homogeneous primal-dual predictor-corrector method handles general
  product cones. It uses Nesterov-Todd scaling for SOC and PSD blocks,
  sparse Schur systems, iterative refinement, and dense PSD operators
  without materializing a full Kronecker matrix.
- Separable exponential epigraphs are reduced to log-sum-exp and exponential
  constraints, solved by a damped primal-dual Newton method with active-set
  refinement. A logarithmic-barrier conic method handles other exponential
  structures.
- Low-rank polyhedral QPs can use fully corrective conditional gradients
  with warm LP oracles. Semidefinite certificate refinement uses affine and
  cone projections. Experimental reduced-space and LDL variants remain
  available in the source but are not the default linear-system path.

Every returned optimum or certificate is checked against the original
input: finite dimensions, scaled primal/dual equations, cone defects,
objective gap, and the contract's normalized certificate tests. An
unverified candidate is reported as `unresolved`. Objective constants are
not reported; the evaluator computes objectives from the original input.
The PSD representation is the specified upper-triangle column ordering
with square-root-of-two off-diagonal scaling.

## Development Evidence

Formal public subset evaluations were used to select numerical paths.
One 131-case checkpoint verified 97 cases, including 8/9 exponential and
6/9 PSD cases. These are single-run development results, not a claim about
held-out results or three-repeat success. Short-limit timing tests favored
the native LP path, sparse equality preprocessing and exponential
epigraph elimination. Numerical experiments that regressed verified cone
cases were not selected as defaults.

## References

- Mehrotra (1992), predictor-corrector primal-dual interior-point methods.
- Nesterov and Todd (1997), self-scaled barriers and symmetric-cone scaling.
- Andersen and Ye (1999), homogeneous interior-point methods.
- Huangfu and Hall (2018), the dual revised simplex implementation used by
  HiGHS; SciPy's bundled HiGHS and SuperLU implementations.
- Nesterov and Nemirovski (1994), logarithmically homogeneous barriers.
- Wolfe (1976), fully corrective minimum-norm-point methods.
