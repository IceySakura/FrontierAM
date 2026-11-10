# Native conic interior-point solver

## How to run

```sh
./build.sh                                  # compiles src/solver.cpp -> bin/solver
./run.sh --input IN.npz --output OUT.npz    # execs bin/solver
```

* `build.sh` takes no arguments and exits 0. If the compile ever fails, it copies
  `bin/solver.prebuilt` to `bin/solver`. That binary was built from the same sources in this
  image.
* `run.sh` sets `OPENBLAS_NUM_THREADS=1` and `exec`s the binary.
* The solver always writes the output NPZ and exits 0, including when the result is
  `unresolved`.
* Diagnostics go to stderr only when `SOLVER_VERBOSE=1` (or 2, 3) is set.
* No temporary files are written.

## Dependencies and pinned versions (all preinstalled; nothing is downloaded)

* **g++ 12.2.0 (C++17):** one translation unit, `src/solver.cpp`, plus two headers:
  * `src/npzio.hpp`: zip/NPY reader and writer.
  * `src/amd.hpp`: approximate minimum degree ordering.
* **zlib 1.2.13 runtime (`libz.so.1`):** inflates the deflate-compressed NPZ inputs. There is
  no `zlib.h` in the image, so the prototypes and the `z_stream` layout are declared in
  `npzio.hpp`.
* **Reference LAPACK/BLAS 3.11 (`liblapack.so.3`, `libblas.so.3`):** used for the small dense
  operations of PSD cones (Cholesky, SVD, symmetric eigenvalues, gemm). When PSD cones are
  present, the solver `dlopen`s the OpenBLAS that ships inside the preinstalled SciPy 1.16.3
  wheel (`scipy.libs/libscipy_openblas-*.so`, symbols `scipy_d*_`) and runs it single-threaded.
  If that library is missing, it silently stays on reference LAPACK.

There is no Python at solve time. Process start-up takes milliseconds. That matters because
most instances have a 1 s limit, and `import numpy, scipy.sparse` alone takes about 0.7 s on
this machine.

## Method

Problem: `min ½x'Px + q'x  s.t.  Ax + s = b, s ∈ K`, with K a product of zero, nonnegative,
second-order, exponential and PSD cones.

The solver is a **homogeneous self-dual embedding interior-point method with a quadratic
objective**, following Clarabel (Goulart & Chen 2024).

* **Presolve and scaling:**
  * Duplicate entries are summed and explicit zeros dropped.
  * Consecutive zero and nonnegative cones are merged.
  * Ruiz equilibration (up to 10 passes, stopping early once the factors have converged;
    factors bounded to [1e-4, 1e4]) is applied to the KKT matrix, with uniform row scaling
    inside each SOC/exp/PSD block, followed by cost scaling.
* **Initial point:**
  * Symmetric cones: the Clarabel/CVXOPT least-squares point (solved with `H = I`), shifted
    into the cone interior.
  * When exponential cones are present: the unit point (central ray of the exp cone).
* **Iteration:** a Mehrotra predictor-corrector on (x, s, z, τ, κ). Each iteration does one
  factorization and three solves (the constant `[-q; b]` system, the affine step and the
  combined step). Δτ is eliminated with the numerically stable denominator
  `-(z1'Hz1 + ‖x1-ξ‖²_P + κ/τ)`. Step fraction is 0.99, and `σ = (1-α_aff)³`.
* **Scalings:**
  * Nonnegative and second-order cones: Nesterov–Todd scaling. The SOC uses a stable residual
    `(x0-‖x1‖)(x0+‖x1‖)`, Clarabel-style renormalisation of `w`, and `W'λ` as the
    affine-step right-hand side.
  * PSD cones: NT scaling from `R = L_s V Λ^{-1/2}`, where `L_z'L_s = UΛV'`.
  * Exponential cones: dual-barrier scaling `μ∇²f*(z)` with centering `s + σμ∇f*(z)`. The step
    length uses backtracking with barrier-domain checks. On the development set this was much
    faster than the Dahl–Andersen primal-dual scaling variant, which is still available with
    `EXP_PD=1`.
* **Linear algebra:**
  * The quasi-definite KKT matrix `[[P+δI, A'], [A, -(H+δI)]]` is ordered with AMD (a port of
    CSparse `cs_amd`) and factored with an up-looking sparse LDLᵀ (the QDLDL algorithm).
  * Static regularization `δ = 1e-8`, plus sign-aware dynamic pivot regularization.
  * Iterative refinement runs against the unregularized matrix. Its relative tolerance is
    `min(1e-9, max(1e-13, 1e-6·μ))` for problems without SOC cones, and 1e-13 otherwise.
  * *Second-order cones are eliminated analytically:* `A_c'H_c⁻¹A_c` is added to the x-block,
    using the exact inverse `H⁻¹ = η⁻²(2ŵŵ' − J)`, `ŵ = Jw`. Near the boundary, the dense
    `-(H+δI)` SOC blocks (condition number ~1e18) produced wrong-sign pivots. The reduced
    solve is then refined against the full unreduced system (up to 10 steps). An SOC whose
    rows touch more than 200 columns keeps its dense `-(H+δI)` block in the KKT instead, which
    prevents a dense |J_c|² fill.
  * *PSD-dominated problems* (only PSD and nonnegative cones, n ≤ 6000) use a dense Schur
    complement (normal equations) `P + Σ A_c'H_c⁻¹A_c` instead of the sparse KKT. Forming it
    exploits the sparsity of each column of `A` via `G_j = B⁻¹ S_j B⁻¹` with `B = RR'`. It is
    then factored with dense Cholesky and the solves are refined.
* **Safeguards:**
  * If a step makes a nearly converged iterate ≥10× worse (or non-finite), the step is undone
    and redone with a different static regularization and a damped step.
  * The best iterate that satisfies the tolerance is kept.
  * Retries with larger regularization run if a factorization fails.
* **Termination:**
  * The independent checker's formulas are evaluated each iteration in the *original*
    coordinates (primal and dual residuals with their normalisations, absolute or relative
    gap) against 0.9·1e-8.
  * Before a result is reported as `solved`, it is re-verified with the original matrices.
  * Primal and dual infeasibility certificates are tested each iteration with the checker's
    normalised conditions (`A/max|A|`, `P/max|P|`, `b/‖b‖`, `q/‖q‖`).
* **Output:** vectors are mapped back to the original coordinates. `s` is exactly 0 on
  zero-cone rows. The objective fields are omitted (they are optional).

## References

* P. J. Goulart, Y. Chen. *Clarabel: An interior-point solver for conic programs with quadratic
  objectives.* arXiv:2405.12762 (2024). This is the HSDE with quadratic objective, the
  Δτ elimination, and the regularization and refinement defaults.
* L. Vandenberghe. *The CVXOPT linear and quadratic cone program solvers* (2010): NT scaling
  formulas for SOC and PSD, the initial point.
* A. Domahidi, E. Chu, S. Boyd. *ECOS: An SOCP solver for embedded systems* (2013), and
  S. Serrano's thesis (2015): exponential-cone dual-barrier scaling and the line search.
* J. Dahl, E. D. Andersen. *A primal-dual interior-point algorithm for nonsymmetric
  exponential-cone optimization.* Math. Prog. (2022). Implemented and evaluated; not the
  default.
* B. Stellato et al. *OSQP*, and the QDLDL factorization. T. Davis, *Direct Methods for Sparse
  Linear Systems* (CSparse `cs_amd`). P. Amestoy, T. Davis, I. Duff, *An approximate minimum
  degree ordering algorithm* (1996).

## Evidence from experiments (public development set, 163 instances, public checker)

Final build: whole-process timing on the shared development core.

* **Overall:** with n+m ≤ 300k, **123 of 147 verified** (122 within the development time
  limits). By group:

  | Group | Verified |
  | --- | --- |
  | exponential | 8/9 |
  | linear | 48/52 |
  | PSD | 6/9 |
  | quadratic | 51/54 |
  | SOC | 10/23 |

* **Larger instances:**
  * The n=155k LP is verified.
  * The six 2–4 million-variable QPs are all verified, in 3.1–15 s against limits of
    3.0–15.7 s, so some of them are borderline.
  * The nine large SOC instances (n ≥ 47k) are not solved.
* **Process start-up:** a Python solver was rejected up front, because importing numpy and
  scipy takes about 0.7 s against mostly 1 s limits.
* **Second-order cones:**
  * With dense SOC blocks inside the quasi-definite LDLᵀ, every SOC instance stalled near
    μ≈1e-7: wrong-sign pivots and `Hz≠s` drift.
  * Analytic SOC elimination plus full-system refinement plus the revert safeguard fixed the
    small and medium SOC instances (17–31 iterations).
* **PSD:** the dense Schur-complement path solves PSD instances with order 50–250 in
  0.04–9 s. The sparse-KKT path took 15 s or failed, and the reference timed out on most of
  them.
* **Exponential cones:** dual scaling roughly halved the iteration counts compared with the
  Dahl–Andersen primal-dual scaling variant (101→47 and 119→36 iterations).
* **Adaptive refinement:** the tolerance is loosened to 1e-9 relative while μ is large, and
  the full 1e-13 is kept near the end and for all SOC problems. This cut 10–30% of solve
  time on LP/QP with no regressions (one extra instance solved).
* **Remaining weaknesses:**
  * Problems whose KKT factor needs about 1e9 or more flops per factorization (the LDLᵀ is not
    supernodal).
  * The largest SOC instances, which lose dual-residual accuracy near the end.
  * One dual-degenerate LP.
