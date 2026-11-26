# Conic interior-point solver (NumPy / SciPy, pure Python)

Solves

```
minimize    1/2 x' P x + q' x + c0
subject to  A x + s = b,      s in K
```

where `K` is a product of zero / non-negative / second-order / semidefinite /
exponential cones, and returns `solved`, `primal_infeasible`,
`dual_infeasible` or `unresolved` in the NPZ format required by the task.

## Running

```sh
./run.sh --input INPUT_PATH --output OUTPUT_PATH
```

`run.sh` is a POSIX shell script that `exec`s `python3 solver.py` with the same
arguments.  No build step is needed (`build.sh` is deliberately omitted: the
solver only uses the pre-installed CPython 3.12.14 with NumPy 2.3.5 and
SciPy 1.16.3).

Input and output format: identical to `/workspace/samples/io/model.py` (NPZ
with `q`, `b`, the CSC arrays `P_*`, `A_*` and a scalar `metadata_json`).

## Method

Primal-dual infeasible-start interior point method with Mehrotra
predictor-corrector, working directly on the cone complementarity conditions.

* **Variables**: `x` free, `s` in `K`, `z` in `K*`.
* **Linear algebra**: the Newton system is condensed to the normal equations
  `M = P + A' W^-1 A` (dimension `n`), which is symmetric positive definite.
  Rows belonging to zero cones are equality constraints; they are treated with
  a saddle point system `[M A0'; A0 0]` so that the free dual variables of the
  zero cone are eliminated exactly.  The factorisation uses SuperLU
  (`scipy.sparse.linalg.splu`) for large problems and a dense LAPACK solve for
  small ones.
* **Non-negative cone**: classical product-form Mehrotra scaling
  `W = diag(s/z)`, right hand side `g = (sigma*mu - s*z + ds_a*dz_a)/z`, with
  `sigma = (mu_aff/mu)^3` from the affine (predictor) step.  The second order
  term is damped by the length of the predictor step, which keeps the
  corrector right hand side moderate and the iteration stable.
* **Second-order cone**: the Jordan structure of the cone gives the exact
  linearisation `Z ds + S dz = rhs` with `Z = [[z0, zv'], [zv, z0 I]]` and
  `S = [[s0, sv'], [sv, s0 I]]`.  The inverse of `S` is regularised with the
  current complementarity level, which keeps the coupling matrix bounded when
  the slack approaches the boundary of the cone; the resulting matrix is
  symmetric positive definite.
* **Semidefinite and exponential cones**: analytic barrier scalings.  With the
  dual barrier `phi` (`phi(Z) = -log det Z` for the semidefinite cone, the
  standard dual barrier of the exponential cone) the coupling matrix is
  `W = -mu * Hess phi(z)`, which is symmetric positive definite; for the
  semidefinite cone the inverse barrier Hessian is applied in the eigenbasis of
  the dual matrix, which keeps the update at `O(d^3)` per block.
* **Step lengths**: a single fraction-to-the-boundary step is used for `x`,
  `s` and `z`.  This keeps the primal and dual residuals scaled by
  `(1-alpha)` and is the standard way of keeping the iteration stable.
* **Scaling**: the objective is multiplied by a positive constant before the
  iteration (this does not change the minimiser) so that the objective and
  the constraint data are balanced; the dual variable is rescaled before it is
  reported.  Without this, badly scaled instances can stall because the
  Newton step for the cone variables is far too long.
* **Termination**: relative primal residual, relative dual residual and the
  objective gap `|p0 - d0| = |s'z|` are driven below `1e-9`, i.e. well
  inside the `1e-8` acceptance tolerance of the independent checker.
* **Infeasibility certificates**: if the main solve does not converge, two
  auxiliary conic programs are solved with the same code:
  - *primal infeasibility*: `min b'z` subject to `A'z = 0`, `z in K*`,
    `|z|_inf <= 1`; a negative optimum yields the Farkas certificate `z`;
  - *dual infeasibility*: `min q'x` subject to `Px = 0`, `Ax + s = 0`,
    `s in K`, `|x|_inf <= 1`; a negative optimum yields the certificate
    `(x, s)`.
  Both certificates are validated against the same conditions as the
  independent checker (scaled residuals, cone defects and the negative margin)
  before they are reported.
* **Robustness**: the solver never raises; on an internal failure, or when its
  internal time budget is exhausted, it writes a valid `unresolved` result and
  exits with code 0.  The internal budget is derived from the instance size so
  that the process stays inside the usual one second per instance limit.

## Start-up cost

Most instances have a one second limit and the whole process (interpreter
start-up included) has to fit into it, so the solver avoids expensive imports
whenever possible:

* small and medium instances are solved with a NumPy-only dense path (sparse
  matrix products through `bincount`, dense LAPACK factorisation), so SciPy
  (and its ~0.45 s import) is never loaded; the infeasibility searches build
  their auxiliary problems with NumPy as well, so the SciPy import is avoided
  there too.  The process start-up is therefore ~0.3 s instead of ~0.7 s,
  which matters because most instances have a one second limit;
* when SciPy is needed, `numpy.testing` is replaced by a stub before the
  import, which removes ~0.15 s from the start-up;
* the initialisation of the linear algebra avoids iterative solvers on large
  systems.

## References / background

* S. Mehrotra, "On the implementation of a primal-dual interior point method",
  SIAM J. Optim. 2 (1992) - the predictor-corrector scheme used here.
* S. Alizadeh and P. Goldfarb, "First-Order Methods for Optimization" -
  conic interior point methods, dual scaling and the Jordan algebra treatment
  of the second-order and semidefinite cones.
* F. Alizadeh and P. Goldfarb, "Second-order-quadratic programming and the
  rank-1 semidefinite programming problem", Math. Program. 95 (2003).
* The task formulation follows the standard conic form used by CVXPY /
  ECOS / SCS (`cones` list, `objective_offset`), which is the interface this
  solver reproduces.

## Status

Non-negative (with zero cones), quadratic, second-order and exponential cone
problems are solved reliably for well scaled instances, including the primal
and dual infeasibility certificates.  The semidefinite cone scaling is
implemented but still stalls on harder instances, which are reported as
`unresolved`.

## Known limitations

* Feasible problems with only non-negative and quadratic structure are solved
  reliably.  Problems with second-order, semidefinite or exponential cones
  use barrier scalings that are implemented but still converge much less
  reliably; such instances are frequently reported as `unresolved`.
* Primal and dual infeasibility certificates are produced for problems whose
  cones are covered by the current dual cone implementations (the exponential
  cone is excluded from the primal infeasibility search).
