# Offline conic QP solver

```sh
./build.sh
./run.sh --input /absolute/path/to/input.npz --output /absolute/path/to/result.npz
```

`build.sh` compiles the C++17 kernels and creates a generic Python bytecode
cache. It does not read any instances. `run.sh` uses one numerical thread,
the inherited CPU affinity, and no network, persistent state, or background
workers. It writes a schema-version-1 NPZ even if the result is `unresolved`.

## Dependencies

- GCC/g++ 12.2; Debian 12 amd64.
- Preinstalled Python 3.12.14, NumPy 2.3.5, and SciPy 1.16.3. The LP path
  uses SciPy's bundled HiGHS 1.8.0. The dense and sparse Newton paths use
  SciPy's LAPACK and SuperLU.
- System `libarchive.so.13` (3.6.2) and `libjansson.so.4` (2.14) for the
  native NPZ entry path. No header packages or additional installs are used.

## Algorithms

The general solver uses a homogeneous primal-dual predictor-corrector
iteration for quadratic objectives. It combines cone-preserving
equilibration, Nesterov-Todd scaling for symmetric cones, and
exponential-cone barrier scaling. Small SOCs and exponential cones use
spectral coordinates in the Newton system to preserve accuracy at the
boundary. Native QP and SOC loops use reusable sparse LDL or dense BLAS
factorization. The general path also has high-fill dense Schur systems
and PSD Schur or QR elimination. Chordal decomposition and explicit-cone
dualization cover additional SDP structures. A proximal splitting fallback
uses a reusable factorization or matrix-free conjugate gradients.

The polyhedral path also includes native bound, diagonal-Hessian and
network-flow solvers for matching structures, HiGHS, and active-set
refinement. Reversible presolve handles singleton equalities, low-fill
substitution, and structural cone faces. Exact independent blocks can be
solved separately, with identical blocks sharing a solve within one input.
All reported solutions and rays
are tested in the original coordinates; objective values are left for the
checker to compute.

## Validation

Development used the five available `samples/inputs` cases, six interface
examples, and generated feasible, infeasible, ill-scaled, and sparse
regressions. A source-only copy was built offline and made read-only
before testing. All eleven bundled checks passed in three fresh-process
runs, as did separate checks of duplicate CSC entries, empty dimensions,
nonpolyhedral infeasibility certificates, and a million-variable box QP.
Large regressions were used to compare bound, sparse, high-fill, and SDP
factorization paths. Public evaluation snapshots are aggregate feedback
only, not inputs to the program.

## References

- Goulart and Chen, *Clarabel: An interior-point solver for conic programs with quadratic objectives*.
- Nesterov and Todd, *Self-scaled barriers and interior-point methods for convex programming*.
- Dahl and Andersen, *A primal-dual interior-point algorithm for nonsymmetric exponential-cone optimization*.
- Stellato et al., *OSQP: An operator splitting solver for quadratic programs*.
- Davis, *Algorithm 849: A concise sparse Cholesky factorization package*.
- Grone et al., *Positive definite completions of partial Hermitian matrices*; Agler et al., *Positive semidefinite matrices with a given sparsity pattern*.
