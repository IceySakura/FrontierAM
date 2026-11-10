# Conic QP solver (homogeneous primal-dual interior point)

Solves `min ½xᵀPx + qᵀx + c₀  s.t.  Ax + s = b, s ∈ K`, with K a product of
zero, nonnegative, second-order, exponential and PSD cones, and returns an
optimal solution or a primal/dual infeasibility certificate (`unresolved`
only if nothing could be verified).

## Running

```
./build.sh                                  # gcc -O3: kkt.c -> libkkt.so (~1 s)
./run.sh --input IN.npz --output OUT.npz    # execs python3 solver.py
```

If `libkkt.so` is missing, `solver.py` compiles `kkt.c` into `$TMPDIR` itself.

Dependencies (all preinstalled in the image, nothing downloaded):
Python 3.12.14, NumPy 2.3.5, SciPy 1.16.3, gcc 12.2.0. Of SciPy only the
SuperLU C extension (`scipy/sparse/linalg/_dsolve/_superlu*.so`) is loaded,
directly by file path: `import scipy.sparse` alone costs ≈0.5 s, while most
instances have a 1 s limit measured from process start.

## Method

* **Algorithm**: homogeneous self-dual embedding with quadratic objective and
  Mehrotra predictor–corrector (as in Clarabel: Goulart & Chen, *Clarabel: An
  interior-point solver for conic programs with quadratic objectives*, 2024).
  Each iteration solves two systems with the quasi-definite KKT matrix
  `[[P+δI, Aᵀ], [A, -(H+δI)]]` (one constant right-hand side, one per
  predictor/corrector) and a scalar equation for Δτ.
* **Scalings**: Nesterov–Todd for nonnegative, second-order and PSD cones
  (Vandenberghe, *The CVXOPT linear and quadratic cone program solvers*);
  exponential cones use the primal-barrier Hessian scaling `(μ∇²f(s))⁻¹`
  with a backtracking line search that keeps every exp cone in a
  neighbourhood (complementarity lower bound plus a fraction-to-boundary rule
  on each log-barrier term, primal and dual).
* **SOC numerics (own modification)**: the dense block `W²` of an SOC is
  numerically indefinite near the boundary (condition number > 1e16), which
  broke the LDLᵀ. The SOC rows are therefore substituted `y = WΔz`: the KKT
  matrix holds `W⁻¹A` in those rows and `-I` in the (2,2) block. For SOC and
  PSD rows the slack step is taken from the linear equation (keeps the primal
  residual exact) and dual-residual correction solves are applied.
* **PSD**: normal equations `Σ (W⁻ᵀA)ᵀ(W⁻ᵀA)` (+ nonnegative rows) with dense
  BLAS, NT scaling via Cholesky + SVD, step lengths computed in the scaled
  space, scaling operators applied sequentially (R…Rᵀ) instead of forming
  `RRᵀ` (better accuracy near the boundary).
* **Sparse linear algebra**: own C kernel (`kkt.c`): up-looking LDLᵀ
  (QDLDL algorithm, Stellato et al.) with static (1e-8) and dynamic
  regularisation, escalating static regularisation if dynamic regularisation
  fires, iterative refinement against the unregularised matrix, O(nnz)
  counting-sort assembly. Ordering: the natural order (or "constraints
  first") is used when a capped symbolic count shows small fill; otherwise
  multiple minimum degree on `K+Kᵀ` from SuperLU (`MMD_AT_PLUS_A`).
* **Preprocessing**: Ruiz equilibration of the KKT matrix (block-uniform row
  scaling for SOC/exp/PSD blocks) and cost scaling.
* **Termination / certificates**: residuals, gap and cone defects are
  evaluated in original coordinates with the published checker formulas
  (with margin) before anything is reported; a side-effect-free polishing
  step (one extra KKT solve on the dual residual) is tried when an iterate is
  almost optimal.
* **Deterministic fallbacks**: if an attempt ends unresolved (and little time
  was used) the solve is repeated with alternative parameter sets (step
  fraction, exp neighbourhood, SOC/PSD variants). Only verified results are
  written, so retries cannot produce wrong answers.

## Experimental notes (public dev set, this machine)

Final regression (each instance one fresh process, dev time limits, independent
checker): about 121 of 163 verified within the dev limit
(exp 7/9, linear 48/53, quadratic 50/60, SOC 14/32, PSD 2/9).
This includes 8 instances whose dev limit is 300 s (where the reference
failed): 2 exp, 5 SOC and 1 PSD. Some more are verified but are too slow.

* Process start + `import numpy` ≈ 0.2 s; tiny instances finish in ≈0.3 s,
  50k-row LPs in ≈0.8 s.
* Using SuperLU as the factorisation itself was ~10x slower than the
  QDLDL-style kernel with the same ordering, so it is only used for ordering.
* Remaining failures: LP/QP instances with very heavy fill (would need AMD
  plus a supernodal LDLᵀ), the 2–4 M variable QPs (≈2x over their limit,
  dominated by the MMD ordering; the largest need >4 GiB here), the largest
  SOC families (numerical breakdown near μ≈1e-9), large PSD instances
  (accuracy of the dense normal equations).
