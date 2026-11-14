# Offline conic QP solver

## Running

```sh
./run.sh --input INPUT.npz --output OUTPUT.npz
```

`run.sh` is a POSIX shell wrapper that sets single-threaded BLAS/OMP
environment variables and `exec`s `python3 -B solver.py`.  No build step is
required (`build.sh` is deliberately absent).  The program always writes a
valid output NPZ, using status `unresolved` if no certificate could be
verified, and exits with status 0.

## Dependencies (fixed versions, all preinstalled)

* Python 3.12.14
* NumPy 2.3.5
* SciPy 1.16.3 (in particular `scipy.sparse`, `scipy.sparse.linalg.splu`
  (SuperLU), `scipy.linalg.cho_factor/cho_solve/solve_triangular`, and
  `scipy.optimize.linprog` with the bundled HiGHS for linear programs)

No network access, no downloads, no compiled extensions of our own, no
temporary files outside `TMPDIR`, and no state shared between invocations.
Everything is single threaded.  Only ordinary files are shipped.

## Method

The problem is

```
min 0.5 x'Px + q'x + c0     s.t.  A x + s = b,  s in K,
                                  z in K*,  P x + q + A'z = 0
```

with `K` a product of zero (equality), nonnegative, second-order,
exponential and positive semidefinite cones.  The implementation is a
self-contained infeasible-start primal-dual interior-point method:

* **Barriers.**  Canonical self-concordant barriers are used:
  `-sum log s` (nonnegative), `-log(t^2-||v||^2)` (SOC),
  `-log det S` (PSD, with the benchmark's `sqrt(2)` svec encoding), and
  `-log(v log(w/v) - u) - log v - log w` (exponential cone, parameter 3).
  Gradient, Hessian-vector products and dense small-block Hessians are
  implemented analytically and unit-tested against finite differences.
  The exponential-cone dual is handled by the checker's linear map
  `T(z) = (-z_2, -z_1, e*z_3)`.

* **Newton system.**  Complementarity is written as `z = -mu grad F(s)`.
  Zero-cone rows are treated as equality constraints with free Lagrange
  multipliers, so each step solves the reduced quasi-definite KKT system

  ```
  [ P + A' W A   A_0' ] [ dx   ] = [ -r_d + A' r_c - A' W r_p ]
  [ A_0          0    ] [ dz_0 ]   [ -r_p0                     ]
  ```

  with `W` the cone scaling.  SuperLU factorizes it once per iteration; the
  affine/predictor and corrector directions share that factorization.
  `W = diag(z/s)` (Nesterov–Todd / Mehrotra) on nonnegative blocks and
  `W = mu grad^2 F(s)` on SOC, exponential and PSD blocks.  PSD blocks use
  the exact identity `W_ij = tr(E_i S^{-1} E_j S^{-1})`, restricted to the
  rows touched by `A` so that large PSD blocks stay cheap.

* **Steps.**  A Mehrotra predictor determines the centering parameter
  `sigma = (1 - alpha_aff)^3`; the corrector adds the second-order term
  `ds_aff*dz_aff/s` on nonnegative blocks.  Step lengths are
  fraction-to-boundary, computed exactly: a ratio test for nonnegative
  cones, the smallest positive root of a quadratic for SOC (including
  `t + alpha*dt > 0`), a generalized symmetric eigenvalue for PSD blocks,
  and a vectorized bisection on the exponential-cone interior conditions.

* **Scaling and polishing.**  Ruiz equilibration of `A` plus global
  balancing of `b`, `x` and the objective keeps all coefficients O(1).
  Every candidate is unscaled and checked with an independent reimplementation
  of the published acceptance conditions; a warm-started polish pass in the
  original coordinates removes scaling-induced residual amplification.
  Several algorithm variants (NT vs. Hessian scaling, common vs. split
  primal/dual steps) are tried with a shared wall-clock deadline.

* **Linear programs.**  When `P = 0` and all cones are zero/nonnegative,
  `lp_fast.py` calls HiGHS through `scipy.optimize.linprog` (fast and exact
  on large sparse LPs).  Dual marginals give the `z` vector.  For
  infeasible/unbounded LPs, auxiliary bounded LPs produce Farkas
  certificates, which are verified before being returned.

* **Infeasibility certificates (general cones).**  When the main solve
  stalls, regularized auxiliary problems are solved with the same IPM:
  `min b'z + rho/2 ||z||^2  s.t. A'z = 0, z in K*` for primal infeasibility
  (the exponential-cone dual is represented by the linear map `T`), and
  `min q'x + 0.5 x'(lambda P + rho I)x  s.t. A x + s = 0, s in K` for dual
  infeasibility.  Certificates are only emitted after passing the published
  normalized checks with margin.

* **Startup.**  A small compatibility shim for `scipy._lib._array_api`
  avoids importing `numpy.testing`/`unittest`/`importlib.metadata` on the
  hot path (saves ~0.23 s per process on the pinned SciPy build).  It lazily
  loads the genuine module if `scipy.optimize` needs the full API.

## Time budget

`run.sh` self-limits each invocation by problem size (roughly 0.6 s of
solver time for small instances, up to ~13 s for very large ones), leaving a
margin under the per-instance limits while including interpreter startup and
imports.  All loops check the deadline and the best verified iterate found
so far is returned.

## References / prior art

* S. J. Wright, *Primal-Dual Interior-Point Methods* (SIAM, 1997) —
  infeasible-start primal-dual framework and Mehrotra predictor-corrector.
* Yu. Nesterov and M. J. Todd, *Primal-dual interior-point methods for
  self-scaled cones*, SIAM J. Optim. 8 (1998) — NT scaling.
* A. Domahidi, E. Chu and S. Boyd, *ECOS: An SOCP solver for embedded
  systems* (ECC 2013) — exponential-cone barrier and HSDE implementation
  ideas.
* P. J. Goulart et al., *Clarabel: An interior-point solver for conic
  programs with quadratic objectives* (2024) — homogeneous embedding and
  quasi-definite KKT formulation.
* B. O'Donoghue et al., *SCS: Splitting Conic Solver* — cone projections and
  infeasibility certificates.
* Skajaa and Ye, *A homogeneous interior-point algorithm for nonsymmetric
  convex conic optimization* (Math. Program., 2015) — Hessian scaling for
  nonsymmetric cones.
* HiGHS / `scipy.optimize.linprog` documentation for the LP path.

Only the algorithmic ideas above were used; all code in this submission was
written from scratch for this task against NumPy/SciPy.

## Development evidence

The public development instances and interface examples were checked with
`python /workspace/samples/io/check_result.py --input ... --result ...`.
The solver verifies small mixed QP, SOC, exponential and PSD examples,
LP/QP instances, and both infeasibility-certificate examples with residuals
below 1e-8.  Larger development cases with many small SOC blocks, many
exponential cones, or a 50 x 50 PSD block are substantially harder; the
implementation tries several variants and polishing passes for them.
