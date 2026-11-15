# Conic solver

## Usage

```sh
./run.sh --input INPUT.npz --output OUTPUT.npz
```

`run.sh` execs `solver.py` (Python 3.12). No build step is required; all code is
plain Python plus NumPy, and every dependency is preinstalled in the evaluation
image. BLAS/LAPACK thread counts are pinned to one in `run.sh`.

* NumPy 2.3.5 is imported normally.
* The HiGHS 1.8.0 shared extension shipped inside SciPy 1.16.3
  (`scipy/optimize/_highspy/_core*.so`) is loaded **directly by path**, so
  `scipy.optimize` (a ~0.35 s import) is avoided on the common path. If that
  file cannot be found the code falls back to importing
  `scipy.optimize._highspy._core`.
* `scipy.sparse` / `scipy.sparse.linalg` are imported lazily, only for large
  instances that need a sparse KKT solve.
* LAPACK `dgetrf`/`dgetrs` are called through `ctypes` from the optimized
  OpenBLAS shipped with SciPy (`scipy.libs`), falling back to system
  `liblapack.so.3`. This gives a reusable LU factorization at near-zero import
  cost.

## Method

The problem is `min .5 x'Px + q'x  s.t.  Ax + s = b, s in K`, with `K` a product
of zero, nonnegative, second-order, PSD and exponential cones.

**Pipeline** (every candidate result is verified against a faithful copy of the
published acceptance conditions before being written; `unresolved` is written
if nothing verifies):

1. **LP / QP** (`K` = zero + nonnegative): HiGHS, invoked through its low-level
   model API with row bounds encoding the two cone types. Primal solutions are
   exact basic solutions; duals come from the row marginals and, when needed,
   are polished by an active-set KKT solve (dense or sparse). HiGHS runs with a
   `time_limit` so a single solve can never monopolise the case budget.

2. **General cones**: an infeasible-start primal-dual interior-point method.
   For each cone, a log-homogeneous barrier `F` supplies the gradient
   `w = -grad F(s)` and a positive definite scaling `H`:
   * nonnegative / SOC / PSD: **Nesterov–Todd scaling** (`H s = z` exactly);
   * exponential cone: `H = mu nabla^2 F(s)`, with a generalized NT variant
     that solves `nabla^2 F(w) s = z` for a scaling point `w`, and a
     primal-barrier path-following fallback. Euler's identity
     (`nabla^2 F(s) s = -grad F(s)`) keeps the nonsymmetric cone consistent.

   The Newton system is reduced either to a quasi-definite saddle system
   `[P+reg, A_0'; A_0, -reg]` in `x` and the zero-cone multipliers (dense case)
   or to `[P, A'; A, -H^{-1}]` (sparse case). Mehrotra affine scaling sets
   `sigma`; fraction-to-boundary step lengths preserve cone interiority.
   Linear systems use a cached, equilibrated LU with iterative refinement
   (LAPACK via ctypes) or SuperLU for large sparse systems. An explicit
   least-norm solution of the zero-cone equalities provides the starting point.

3. **Infeasibility certificates**: a primal-infeasible direction `z in K*`
   with `A'z = 0, b'z < 0`, or a dual-infeasible direction `x` with `Px = 0`,
   `-Ax in K`, `q'x < 0`. HiGHS auxiliary LPs handle the polyhedral case;
   auxiliary conic problems handle the general case; in addition, diverging
   interior-point iterates are tested directly as certificates (the iterates
   naturally converge to Farkas / dual rays). Every certificate is checked with
   the published scaling before it is reported.

A valid `unresolved` result is written immediately at startup (atomic rename),
so even a process killed by a short per-instance time limit leaves a
well-formed NPZ. A daemon watchdog thread bounds the total work per case.
Computation is deterministic and single-threaded.

## Files

| file | purpose |
| --- | --- |
| `run.sh` | executable entry point |
| `solver.py` | dispatch, deadlines, verification, output |
| `nputil.py` | CSC sparse arrays (NumPy only), NPZ I/O |
| `verify.py` | internal copy of the independent checker |
| `highs.py` | direct HiGHS wrapper, QP/LP path, KKT polish |
| `ipm.py` | dense primal-dual / barrier interior-point solvers, cone algebra |
| `ipm_sparse.py` | sparse quasi-definite and normal-equations IPM |
| `linsolve.py` | reusable LU via LAPACK/ctypes |
| `certs.py` | auxiliary problems for infeasibility certificates |

## References

* S. J. Wright, *Primal-Dual Interior-Point Methods*, SIAM 1997.
* Yu. Nesterov and M. J. Todd, "Self-scaled barriers and interior-point
  methods for convex programming", Math. Oper. Res. 22 (1997).
* A. Domahidi, E. Chu, S. Boyd, "ECOS: An SOCP solver for embedded systems",
  ECC 2013 (SOC NT scalings).
* P. Goulart et al., "Clarabel: An interior-point solver for conic programs
  with quadratic objectives", 2024.
* A. Skajaa and Y. Ye, "A homogeneous interior-point algorithm for
  nonsymmetric convex conic optimization", Math. Program. 150 (2015).
* HiGHS 1.8.0 (https://highs.dev), bundled with SciPy 1.16.3.
