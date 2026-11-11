# Conic solver (QP objective, zero / nonnegative / SOC / exponential / PSD cones)

## Running

```sh
./run.sh --input INPUT.npz --output OUTPUT.npz
```

`run.sh` uses `exec` to start `python3 solver.py` with the same arguments. No build step is needed,
so there is no `build.sh`. Single-threaded BLAS is enforced through `OPENBLAS_NUM_THREADS=1` and
related variables. Optional environment knobs, all of which have defaults:
`SOLVER_TIME_CAP` (seconds, default 290), `SOLVER_VERBOSE=1` (iteration log on stderr).

## Dependencies (pinned, all preinstalled in the image)

* Python 3.12.14
* NumPy 2.3.5 (OpenBLAS 0.3.30)
* SciPy 1.16.3 (SuperLU sparse LU, LAPACK)

No other packages or network access are used.

## Method

The solver is a primal-dual interior-point method on the homogeneous self-dual embedding (HSDE)
for quadratic objectives. It follows the Clarabel formulation (Goulart & Chen, 2024):

* Residuals: `r_x = Px + A'z + q tau`, `r_z = Ax + s - b tau`,
  `r_tau = kappa + q'x + b'z + x'Px/tau`.
  The Newton step is computed with the two-solve trick: one solve with the constant right-hand
  side `[-q; b]`, then one solve per predictor/corrector step. `dtau` is obtained in closed form.
* Mehrotra predictor-corrector with `sigma = (1 - alpha_aff)^3` and a 0.99 fraction-to-boundary
  rule.
* Symmetric cones (nonnegative, SOC, PSD) use Nesterov-Todd scaling (CVXOPT/ECOS formulas; for
  PSD, Cholesky factors of S and Z plus an SVD), with the Mehrotra second-order correction in the
  Jordan algebra.
* Exponential cone (nonsymmetric) uses the primal-dual scaling of Dahl & Andersen (2022) and
  Tuncel: a rank-4 BFGS-type update of `mu * Hess f*(z)` that maps both `z` to `s` and the
  shadow iterates `z~` to `s~`. The primal gradient is the conjugate barrier gradient, computed
  through the Wright omega function. The update falls back to dual scaling near the central
  path. The corrector uses the third-order term `-1/2 f*'''(z)[dz, Hess^{-1} ds]` (closed form).
  Step lengths come from backtracking on strict primal/dual exp-cone feasibility.
* Initialisation: Clarabel-style for symmetric cones (one KKT solve with `H = I`, then a shift
  into the cone). Unit/central-point initialisation is used when exponential cones are present.
* Preprocessing: modified Ruiz equilibration of `[P A'; A 0]`. Scaling factors are made uniform
  inside SOC, exponential and PSD blocks so that cone membership is preserved. The cost is also
  scaled. All vectors are mapped back to the original coordinates and row order.

### Linear algebra

* **Sparse mode** (default): the quasi-definite KKT matrix `[P+dI, A'; A, -(H+dI)]` is assembled
  once. Only its numerical values are updated in place each iteration: diagonal for LP rows,
  dense small blocks for SOC/exp cones. It is factorised with SuperLU (`MMD_AT_PLUS_A` ordering,
  symmetric mode, pure diagonal pivots by default), followed by up to 10 steps of iterative
  refinement against the unregularised system. See "Adaptive pivoting" below.
  Experiment: on the public LP `dev-0fd6404fc8d3`, pure diagonal pivoting with `d = 1e-8` stalled
  at a dual residual of about 5e-7. Switching to threshold 0.1 with `d = 1e-9` converged in 30
  iterations.
* **Dense mode** (any PSD block with more than 60 svec entries and `n <= 8000`): normal equations
  `M = P + A' H^{-1} A`. The PSD part `H^{-1} = G^{-1} (x)_s G^{-1}` is formed only on the svec
  rows that `A` actually touches. `M` is factorised with dense Cholesky, zero-cone rows are kept as
  a Schur complement, and iterative refinement is again done against the true KKT system. In this
  mode the slack step is taken from the linearised primal equation, which keeps primal
  feasibility accurate when the NT scaling is ill-conditioned.
  Experiment: for the d=50 public PSD instance, the sparse KKT took 30 s and dense mode 0.8 s.

### Numerical safeguards

These were added after failures on the public development set:

* **Monotone iterative refinement.** A correction is kept only if it reduces the residual of the
  true (unregularised) KKT system. Previously, refinement on an inaccurate factorisation diverged
  to directions of size 1e60 on SOC problems.
* **Proportional block regularisation.** Each dense SOC/exp/PSD block of `H` gets
  `1e-9 + 1e-11 * mean(diag(H_k))` on its diagonal. This emulates the dynamic regularisation of
  LDL solvers: near the cone boundary, the SOC scaling blocks `eta^2 (2ww' - J)` have condition
  numbers around 1e16, and elimination with diagonal pivots produced wrong-sign pivots.
  Refinement removes the perturbation.
* **Adaptive regularisation.** If the constant-RHS solve still has relative residual > 1e-6, the
  static regularisation is raised (x100, up to 1e-5) and the matrix is refactorised.
* **Pivoting.** SuperLU uses pure diagonal pivots by default; with the safeguards above this was
  the fastest and most robust setting on the public set. For example, the SOC instance
  `dev-165f6362595c` takes 71 iterations and 1.7 s without pivoting, against 116 iterations and
  10 s with adaptive threshold pivoting. Threshold pivoting (0.1) is kept only in the retry
  cascade below, for small systems (`n+m <= 20000`). It is switched off permanently if a pivoted
  factorisation costs more than 20x the unpivoted one: on a public LP with 66k rows, threshold
  pivoting increased the LU fill from 0.19M to 4.5M entries and the factorisation time from
  0.07 s to 10 s.
* **Neighbourhood safeguard.** After the fraction-to-boundary rule, the step is backtracked (x0.8)
  until the worst nonnegative/SOC pair centrality `sqrt(det(s) det(z))`, together with `tau*kappa`,
  is at least `min(1e-4 * mu_new, 0.5 * current value)`. This keeps iterates from collapsing onto
  the cone boundary. The relative part avoids freezing iterates that start outside the
  neighbourhood: QP `dev-044498a5c889` stalled before this was added and is now solved in 0.8 s.
* **Accurate dense PSD path.** In dense mode the solver switches from Cholesky of
  `M = A' H^-1 A` to a QR factorisation of `B = [W^-T A_psd; sqrt(z/s) A_nn; sqrt(d) I]` once
  solves lose accuracy (relative residual > 1e-6), so the conditioning is not squared. This path is used only when `P = 0`
  and there are no SOC/exp cones.
* **Retry cascade.** If a run ends unresolved within the first 30% of the time cap, the solve is
  repeated, first with adaptive threshold pivoting (switched on after three short steps or an
  inaccurate solve), then with threshold pivoting from the start plus regularisation 1e-7. This
  recovers, for example, the SOC instance `dev-4a42fbac0fcf`.

### Termination and certificates

Each iterate is unscaled and checked with a replica of the official acceptance tests: relative
primal and dual residuals, absolute-or-relative gap, and cone membership (strict interiority). A
result is returned as `solved` when all three measures are at most `5e-9`. When the final iterate
fails, the best iterate seen that passes `1e-8` is returned instead. Primal and dual infeasibility
are declared only when the normalised certificate (`z`, or `(x, s)`) passes the official scaled
certificate tests with margin. Otherwise the status is `unresolved`, and an output file is always
written. The objective values are not reported, which the output format allows.

### Development results (public set, local machine, single core)

Checked with `public-dev/io/check_result.py`. A sample of the 163 public instances was run with
a 40-100 s cap per instance. Timings include about 0.7 s of Python/NumPy/SciPy start-up.

| group | verified / run | notes |
| --- | --- | --- |
| interface examples | 6 / 6 | about 0.75 s each |
| linear | 15 / 15 (incl. 3 primal-infeasible) | 0.75 s to 12 s (`dev-613a8c492d17`, 114k rows) |
| quadratic | 21 / 22 | `dev-cf5401c6abce` fails: LU fill about 10M for every ordering. Instances with n of 2-4M were not attempted. |
| exponential | 9 / 9 | 0.8 s to 40 s (122k rows) |
| psd | 6 / 9 | 0.9 s to 19 s (d = 250). `dev-2ac914d7aa25`, `dev-75c93e958597`, `dev-ca1003b3fb46` stall at a relative dual residual of about 1e-6 (normal-equation accuracy). |
| soc | 5 / 12 run | instances with m = 1576 and m = 9168 solve (1.7-3.3 s), except `dev-8b4fb2c9dddf`. Medium/large SOC (m >= 13k) still stall or exceed 60-100 s. This is the main known weakness. |

## References

* P. Goulart, Y. Chen, *Clarabel: An interior-point solver for conic programs with quadratic
  objectives*, 2024. HSDE with quadratic objective, initialisation, step logic.
* J. Dahl, E. Andersen, *A primal-dual interior-point algorithm for nonsymmetric exponential-cone
  optimization*, Math. Prog. 2022. Exp-cone scaling and the third-order corrector.
* L. Vandenberghe, *The CVXOPT linear and quadratic cone program solvers*, 2010. NT scaling for
  SOC/PSD.
* A. Domahidi, E. Chu, S. Boyd, *ECOS: An SOCP solver for embedded systems*, 2013.
* X. S. Li, *SuperLU*. Sparse LU used through SciPy.
