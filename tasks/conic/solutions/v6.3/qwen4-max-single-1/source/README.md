# Conic solver (offline, single-file C)

## Run

```sh
./build.sh                 # compiles ./solver with gcc -O2 (no external deps)
./run.sh --input IN.npz --output OUT.npz
```

`run.sh` `exec`s the native `solver` binary. Only the C standard library and
`libm` are used; NumPy/SciPy are NOT required at run time. The NPZ (zip +
deflate + npy) reader/writer and a small JSON parser are implemented in C
(`npzio.c`, `inflate.c`), so process startup is a few milliseconds.

## Method

Primal-dual Mehrotra predictor-corrector interior-point method for
`min 1/2 x'Px + q'x + c0  s.t.  Ax + s = b, s in K`, with K a product of
zero, nonnegative, second-order, exponential and semidefinite cones
(Clarabel-style `svec` encoding, off-diagonals scaled by sqrt(2)).

* Sparse KKT system `[P+d1 I, A'; A, -(H+d2 I)]` is factorized once per
  iteration with a custom minimum-degree ordering + sparse LDL^T
  (`kkt.c`), with one step of iterative refinement. SOC blocks use the
  Nesterov-Todd scaling expressed as diagonal + rank-1 (one auxiliary
  variable per block); exponential blocks use the inverse barrier Hessian
  scaling; PSD blocks use the NT congruence scaling with dense symmetric
  eigendecompositions (cyclic Jacobi, `dense.c`).
* For problems with a quadratic term (P != 0) a direct primal-dual IPM is
  used. For linear/conic problems (P = 0) a homogeneous self-dual embedding
  (tau/kappa) is used, which additionally yields primal/dual infeasibility
  certificates.
* Termination tests mirror the independent checker exactly (relative
  primal/dual residuals, per-block normalized cone defects, absolute or
  relative objective gap, and the scaled certificate conditions) at a
  slightly tighter internal tolerance (8e-9 vs 1e-8).

## References

* P. Goulart, Y. Chen, "Clarabel: An interior-point solver for conic
  programs with quadratic objectives" (2024) — problem format, NT scaling,
  KKT structure and homogeneous embedding.
* B. O'Donoghue et al., "Conic optimization via operator splitting and
  homogeneous self-dual embedding" (SCS, 2016) — certificate conditions.
* L. Vandenberghe, "The CVXOPT linear and quadratic cone program solvers"
  — Nesterov-Todd scaling for the second-order and semidefinite cones.
* T. Davis, "Direct Methods for Sparse Linear Systems" — sparse LDL^T.

## Files

`ipm.c` (IPM + cones + assembly + termination), `kkt.c/.h` (ordering, LDL),
`dense.c` (Jacobi eigensolver, matmul), `npzio.c/.h` (NPZ/JSON IO),
`inflate.c` (raw DEFLATE), `common.h`.

## Status / limitations

The zero/nonnegative (LP/QP) path is validated against the public checker,
including primal- and dual-infeasibility certificates. SOC/exponential/PSD
scalings are implemented but not yet fully tuned; those blocks may return
`unresolved` on some instances. The solver is single-threaded and
deterministic, writes a result for every input, and exits 0.
