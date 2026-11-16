# Conic solver (HSDE interior point, native C)

Solves `min ½xᵀPx + qᵀx + c0  s.t.  Ax + s = b, s ∈ K` with K a product of zero,
nonnegative, second-order, exponential and PSD cones (NPZ input/output as specified).

## Usage
```
./build.sh                      # optional: rebuilds bin/solver (a prebuilt binary is shipped)
./run.sh --input IN.npz --output OUT.npz
```
* `run.sh` sets `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` and `exec`s `bin/solver`.
* Build dependencies: gcc 12.2 (Debian 12), system reference LAPACK/BLAS 3.11
  (`/usr/lib/x86_64-linux-gnu/liblapack.so.3`, `libblas.so.3`; used for small dense PSD-block
  operations: `dpotrf`, `dgesvd`, `dsyev`), libm, libdl. If compilation fails, `build.sh` keeps
  the shipped prebuilt `bin/solver` and exits 0.
* Run-time optional dependency: the single-threaded OpenBLAS 0.3.30 shipped inside the
  installed SciPy 1.16.3 wheel (`site-packages/scipy.libs/libscipy_openblas*.so`), resolved
  with `dlopen` at start-up (2–4 ms) and used for `dgemm`/`dtrsm` in dense supernode updates
  and PSD matrix products. If it is absent, built-in loops are used (same results up to
  rounding).
* No Python at run time: on this machine Python start-up plus `import numpy` costs ≈0.2 s and
  `import scipy.sparse` ≈0.6 s, which matters with 1-second limits. The process is
  single-threaded, deterministic, and writes `unresolved` on any internal failure.
* Debug output: `CONIC_VERBOSE=1|2|3` (stderr).

## Method
* **Algorithm**: homogeneous self-dual embedding for QP-objective conic programs with a
  Mehrotra predictor–corrector, following Clarabel (Goulart & Chen, *Clarabel: an
  interior-point solver for conic programs with quadratic objectives*, 2024). The iterate is
  checked every iteration against the evaluator's exact acceptance tests in original
  coordinates (relative residuals, gap, cone defects, primal/dual infeasibility certificates
  with the specified normalisations), with a 10% safety margin, so a status other than
  `unresolved` is only returned when the independent check is expected to pass.
* **Cone scalings**: Nesterov–Todd for nonnegative, SOC and PSD cones; for exponential cones
  the Dahl–Andersen primal–dual scaling (rank-3 BFGS-type update of μ∇²f*(z), conjugate
  barrier gradient via the Wright ω function) with fallback to dual scaling, plus the
  third-order corrector (Dahl & Andersen, *A primal-dual interior-point algorithm for
  nonsymmetric exponential-cone optimization*, Math. Prog. 2022). All exp-cone derivative
  formulas were verified numerically against finite differences.
* **KKT system**: quasi-definite `[P+εI, Aᵀ; A, −(H+εI)]`.
  - SOC blocks always use a *quasi-definite* sparse expansion `H = η²(D + uuᵀ − vvᵀ)` with two
    extra variables per cone, parametrised so that `D − vvᵀ ≻ 0` (extra variables with signs
    +1/−1). Dense SOC blocks caused wrong-sign pivots near convergence; the expansion removed
    them (public SOC sample: failure → 20 iterations).
  - PSD blocks use the *scaled form*: rows `W⁻ᵀA` (computed with sparse congruences) and a
    `−I` block, so the factorisation forms `P + AᵀH⁻¹A` implicitly instead of factoring the
    explicit, extremely ill-conditioned block `H = M⊗M`; iterative refinement is done on the
    original reduced system. (Public PSD sample, which the reference does not solve:
    11 s/failure → 0.3 s/solved.)
  - Δs is recovered from the KKT's own representation of H (SOC expansion variables; primal
    equation for PSD) to avoid catastrophic cancellation; the Δτ elimination uses the formula
    consistent with the computed (regularised) solves.
* **Sparse LDLᵀ** (`src/ldl.c`): own port of AMD (Amestoy–Davis–Duff, dense-row
  postponement) with a work-budget fallback (if moderately dense rows make the element lists
  grow quadratically, it re-orders once with a lower dense threshold; Lasso-type KKT:
  0.56 s → 0.034 s, identical fill on all other test matrices), elimination-tree postordering, relaxed (CHOLMOD-style) supernode
  amalgamation and a left-looking **supernodal** LDLᵀ with BLAS-3 updates; QDLDL-style
  up-looking kernel when the factor has little fill. Static regularisation ε=1e-8,
  sign-aware dynamic regularisation applied pivot by pivot (1e-13 / 2e-7), iterative
  refinement with an adaptive target (10⁻³ of the current linear residuals on the first
  attempt, strict 1e-13 on retries). Measured here: dense 2000² Schur-complement factor
  1.79 s → 0.11 s; random QP (n=2000) 34 s → 1.2 s; 50k-cone SOCP 40 s → 6 s.
* **Robustness**: if an attempt ends unresolved it is restarted with static regularisation
  1e-7, then (exp cones) with dual scaling, then 1e-10 (these fix factorisation orderings that
  are numerically unstable for ε=1e-8). The exp-cone switch to dual scaling only happens when
  the exp-cone backtracking (not another cone) limited the step.
* **Preprocessing**: duplicate summation, removal of nonnegative rows with `b ≥ 1e20`
  (restored as `s = b − Ax`, `z = 0`; automatic re-run without presolve if the reduced problem
  turns out unbounded), Ruiz equilibration (10 passes, cone-block-uniform row scaling for
  SOC/EXP/PSD), cost scaling.
* **I/O**: NPZ reading/writing in C (own DEFLATE decoder, CRC-32, ZIP64 parsing; output uses
  stored members).

## Experimental basis
Developed against the 5 public samples, the 6 interface examples, ~60 generated problems
(LP/QP incl. rank-deficient, badly scaled, degenerate assignment, PDE-control (CONT-like),
Lasso/Huber/SVM/MPC/random OSQP-style QPs, infeasible and unbounded cases; SOCPs with one
3001-dim cone and with 50k small cones; exp-cone logistic regression, entropy, relative
entropy, log-sum-exp, closure-boundary solutions; single/multi-block SDPs, max-cut SDPs), and
~1500 random mixed-cone fuzz problems, all verified with the public checker; an independent
ASan/UBSan review run; plus the public-set feedback runs (55.9 → 82.4 during development;
final configuration 80.9, with timing noise of about ±1 case). A stall detector (stop after 30
iterations without progress) was tried and removed: it cut off slowly converging SOC
instances on the public set (19/19 → 14/19).
Known weak spots: weakly infeasible problems, and extremely ill-conditioned KKT systems
(e.g. projections onto convex sequences with n ≈ 10⁴, where `D₂D₂ᵀ` has condition ≈ n⁴).

## Files
`run.sh`, `build.sh`, `bin/solver` (prebuilt), `src/solver.c` (IPM, cones, KKT, presolve,
checks), `src/ldl.c`/`ldl.h` (AMD + supernodal LDLᵀ), `src/npzio.c`/`npzio.h` (NPZ I/O).
