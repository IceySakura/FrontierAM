# Offline Conic Solver

## Build And Run

```sh
./run.sh --input /absolute/input.npz --output /absolute/output.npz
```

The submission includes the Linux-amd64 `ldl.so` helper, built from the
included `ldl.cpp`. No build or installation is required. The optional
`./rebuild.sh` reproduces the helper and prepares Python bytecode without
downloads or agent tools. Runtime dependencies
are the supplied Python 3.12.14, NumPy 2.3.5, SciPy 1.16.3, and GCC/G++
12.2.0's standard runtime; G++ itself is needed only to rebuild.
The installed SciPy distribution supplies HiGHS 1.8.0, SuperLU,
sparse kernels, and LAPACK. The native adapters use those pinned extension
interfaces directly to avoid eager Python-package imports.

All numerical work is single-threaded. CPU affinity is inherited, not
hard-coded. Solving does not modify the input or submission directory,
use network access, or retain state between instances. Results use the
specified NPZ schema; optional objective fields are omitted.

## Methods

- NumPy-only checks and exact separable LP/diagonal-QP solutions.
- Scalar-dual resource allocation for positive diagonal QPs with box
  bounds and one equality.
- Structural cone reductions, fixed-coordinate compression, and sparse
  equality elimination, with primal and adjoint dual recovery.
- Direct equality-QP solves and a reduced Newton method for monotone
  exponential epigraphs (geometric-program sections).
- Checked least-squares reduction for single norm epigraphs, and
  low-rank Schur systems for large SOC blocks.
- HiGHS simplex/interior-point and active-set QP methods, selected using
  dimensions and sparsity. An active-set KKT polish improves QP accuracy.
- Homogeneous primal-dual interior-point iterations for general conic
  QPs, with Mehrotra steps and Nesterov-Todd SOC/PSD scaling.
- A degree-three exponential-cone barrier, factored Hessians, secant
  scaling, and a central-neighborhood line search.
- Reusable sparse LDL with dynamic regularization, SuperLU fallback,
  diagonal/bound-QP Schur complements, and dense Cholesky where useful.
  PSD blocks use scaled QR solves to avoid squared conditioning.
- Cached native CSC assembly, fused residual estimation, and
  barrier-dependent iterative-refinement accuracy for large sparse cases.
- Ruiz equilibration, iterative refinement, and original-coordinate
  checks of solutions and both kinds of infeasibility certificate.

`entry.py` implements input/output and lightweight presolve; `solver.py`
contains the general algorithms; `gp.py` handles exponential epigraph
reduction; `fastsparse.py`, `fastla.py`, and `kernels.py`
adapt installed numerical kernels; `native_ldl.py` wraps `ldl.cpp`.
`soc_ls.py` supplies the norm reduction; `native_kkt.py` and
`native_state.py` provide the native sparse assembly/residual adapters.
`single_eq.py` supplies the scalar-dual QP path.

## Development Checks

The supplied five development inputs and six interface examples pass the
independent checker. Generated optimality tests also cover all cone types,
mixed quadratic objectives, sparse/dense systems, and degenerate faces.
Method selection was informed by interleaved fresh-process comparisons,
not input names or stored solutions. Development timing is approximate;
the independent evaluator remains authoritative.

## References

- Nesterov and Todd, *Self-scaled barriers and interior-point methods for
  convex programming* (1997).
- Mehrotra, *On the implementation of a primal-dual interior point method*
  (1992).
- Andersen, Dahl, and Vandenberghe, *Implementation of nonsymmetric
  interior-point methods for linear optimization over sparse matrix
  cones* (2010).
- Davis, *Direct Methods for Sparse Linear Systems*, SIAM (2006).
- Huangfu and Hall, *Parallelizing the dual revised simplex method* (2018).
- The installed NumPy, SciPy, SuperLU, LAPACK, and HiGHS implementations.
