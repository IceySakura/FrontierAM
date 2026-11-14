# Conic solver (LP / QP / SOCP / ExpCP / SDP)

Solves

    min_{x,s}  1/2 x'Px + q'x + c0     s.t.  Ax + s = b,  s in K

where `K` is a product of zero, nonnegative, second-order (`soc`), exponential and
positive-semidefinite (`psd`) cones, and returns `solved`, `primal_infeasible`,
`dual_infeasible` or `unresolved` in the NPZ format described in the task.

## Run

    ./build.sh                    # optional, compiles libfast.so (~1 s)
    ./run.sh --input IN.npz --output OUT.npz

`build.sh` needs only a C compiler; if it is missing or fails, `solver_core.py`
transparently falls back to pure numpy implementations of the same kernels.
Run-time dependencies: python 3.12, numpy 2.3.5 (LAPACK for the factorisations and
eigendecompositions).  **scipy is deliberately not imported**: the whole timing
window of a 1-second instance is otherwise consumed by interpreter/library start-up
(measured here: `import numpy` 0.19 s vs `import scipy.sparse` 0.59 s).

Files: `run.sh` (entry point, single-threaded BLAS env), `solver.py` (I/O, status
decision, certificates), `solver_core.py` (algorithm), `fast.c`/`libfast.so`
(sparse matvec + Schur-complement assembly kernels).

## Method

Primal-dual path-following interior point method with a Mehrotra-type
predictor-corrector step, applied to the self-concordant barrier of each cone:

| cone | barrier phi | nu |
| --- | --- | --- |
| nonnegative | `-sum log s_i` | 1/row |
| soc | `-log(t^2 - |v|^2)` | 2 |
| exponential | `-log v - log w - log(v log(w/v) - u)` | 3 |
| psd | `-logdet S` | order |
| zero | (equality, no barrier) | 0 |

The exponential barrier is the aggregate barrier of the three concave inequalities
`v>0`, `w>0`, `v log(w/v) - u > 0` whose domain closure is exactly `K_exp`; its
gradient and Hessian are available in closed form, so the exponential cone is
handled by the *same* Newton scheme as the symmetric cones instead of needing a
first-order fallback.

One damped Newton step per iteration solves

    F1 = Px + q + A'z = 0,  F2 = Ax + s - b = 0,  F3 = z + mu grad phi(s) = 0

Reducing the 3x3 KKT system gives the (n + n_zero) saddle point system

    [ P + A_c' (mu H) A_c   A_z' ] [ dx  ]
    [ A_z                    0   ] [ dz_z]

which is eliminated in two stages (Cholesky/LU of the positive definite `(1,1)`
block, then of the small `A_z M^-1 A_z'` Schur complement).  `M` is assembled
*densely* but *sparsely*: `schur_diag`/`schur_gen` in `fast.c` accumulate
`sum_r w_r a_r a_r'` and `A_blk' H_blk A_blk` with cost proportional to the
nonzeros squared per row, not to `n^2 m`.  PSD blocks use the classical
`Y_j' (S^-1 Y_l S^-1)` formulation with pre-extracted dense column blocks.

Robustness measures that turned out to matter on the development instances:

* rows are permuted so that every cone family is a contiguous block range and all
  blocks of one family are batched into a single vectorised numpy call;
* Ruiz-type equilibration of `A`, `P`, `q`, `b`, with the row scaling forced to be
  **uniform inside a cone block** (a positive uniform rescaling leaves every one of
  the five cones invariant, so the transformation is exact) plus an objective scale;
* `mu` is taken from the iterate itself (`<s,z>/nu`) so the Newton system stays
  consistent with `z ~ -mu grad phi(s)`, while the *target* `mu_t` is floored by the
  current infeasibility level `|F1||F2|/nu`; without that floor `mu` collapses while
  the residuals are still large and the iterates stick to the cone boundary;
* step length is the exact boundary distance per cone (quadratic root for `soc`,
  bisection on the concave function for `exp`, generalised eigenvalue for `psd`),
  times `tau = 0.98`;
* stopping test is the acceptance criterion itself, evaluated in the *original*
  coordinates, at 1e-10 (a 100x margin below the required 1e-8); the best iterate is
  kept, so a late round-off blow-up cannot cost the instance.

## Status decision

`solved` is only reported when the independently re-computed residuals
(primal, dual, both cone defects, absolute/relative duality gap) are all below
1e-9 in the original coordinates.  If the IPM fails, the last iterate is tested
(and polished by alternating projections) as a Farkas certificate for
`primal_infeasible`; otherwise `unresolved` is written.  An output file is always
produced, and every array is finite-checked before writing.

## References

* Mehrotra, S. "On the implementation of a primal-dual interior point method for
  linear programming." SIAM J. Optim. 2 (1992).
* Nesterov, Y., Nemirovskii, A. "Interior-Point Polynomial Algorithms in Convex
  Programming." SIAM, 1994 (self-concordant barriers, `-log` of a concave function).
* Andersen, Dahl, Tunçel/`[TNB08]` combined-step predictor-corrector scheme for
  non-symmetric cones, as used by Clarabel and ECOS.
* Boyd, S. "Notes on the primal-dual interior-point method" (infeasible-start μ).
* Alizadeh, Hoberg. Logarithmic barriers and the second-order/exponential cone.
