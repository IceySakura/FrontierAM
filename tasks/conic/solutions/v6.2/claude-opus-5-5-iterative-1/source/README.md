# Conic solver (HSDE interior point, native C++)

## Running
```
./build.sh                                   # compiles ./solver (g++ -O2, ~5 s)
./run.sh --input IN.npz --output OUT.npz     # exec's ./solver
```
`run.sh` uses `exec` to start the native binary. No Python at solve time: importing
numpy+scipy alone costs ~0.7 s, which would make the 1-second instances impossible.

## Dependencies (all preinstalled in the image, nothing downloaded)
* g++ 12.2.0 (C++17)
* system zlib `libz.so.1` (1.2.13), used only to inflate the NPZ input; declared
  manually because no zlib headers are installed
* reference LAPACK/BLAS (`liblapack.so.3` 3.11.0, `libblas.so.3`), used only for PSD cones
  (`dpotrf`, `dgesvd`, `dsyev`)

## Method
* **Formulation**: homogeneous self-dual embedding for QP objectives, following Clarabel
  (Goulart & Chen, "Clarabel: an interior-point solver for conic programs with quadratic
  objectives", 2024). Predictor-corrector with σ = (1-α_aff)^3 and a Mehrotra
  second-order correction for symmetric cones. Uses a 0.99 fraction to the boundary.
* **Scaling**: Nesterov–Todd for nonnegative, SOC (Lorentz hyperbolic-rotation form as in
  ECOS/Clarabel) and PSD (chol+SVD form as in CVXOPT). Exponential cone uses dual scaling
  μ∇²F*(z) with the explicit dual barrier, plus a barrier-based neighbourhood line search
  (Skajaa–Ye / ECOS-exp style).
* **Linear algebra**: own up-looking sparse LDLᵀ (QDLDL algorithm, Stellato et al.)
  with static (1e-8) and sign-aware dynamic regularization, an AMD fill-reducing ordering
  (re-implementation of CSparse `cs_amd`, T. Davis), and iterative refinement.
* **Improvement over the plain Clarabel KKT for SOC**: near the optimum the NT block
  η²(2wwᵀ−J) loses its small eigenvalue in floating point (w₀ grows like μ^-½), giving
  wrong-sign pivots. For SOC blocks of dimension ≤ 16 the factorized matrix uses the
  W-scaled form [P, (W⁻¹A)ᵀ; W⁻¹A, −I]. That matrix is used only as a preconditioner:
  iterative refinement runs against the original KKT system, with H applied as W(W·),
  because K = diag(I,W)·K'·diag(I,W). Δs on SOC rows is taken from the linearized
  primal equation, which keeps primal residuals consistent. Experimental basis: on the
  public SOC instances this changed every case from stalling at about 1e-8 to verified.
* **PSD problems (nonnegative + PSD cones only, n ≤ 6000)**: these problems use a reduced
  normal-equations solve instead of the dense svec KKT blocks. The solve forms
  M = P + δI + AᵀH⁻¹A, with H⁻¹(X) = QXQ and Q = R⁻¹ from the NT scaling. M is
  assembled entry by entry from pairs of A's PSD nonzeros
  (H⁻¹[(ij),(kl)] = s_ij s_kl (Q_ik Q_jl + Q_il Q_jk)/2), then factored with dense Cholesky,
  as in classical SDP codes (SDPA/CSDP). This is used as the preconditioner of the same
  iterative refinement. Experimental basis: public PSD orders 100, 150 and 200 solve in
  1.1 s, 5 s and 19 s. Previously they were impossible: a dense 11k–20k block in the
  sparse LDLᵀ.
* **Exponential cone line search**: if the centrality backtracking cannot find an
  acceptable step (the current point already lies outside the neighbourhood), the solver
  takes 0.9× the feasible step instead of stalling. On the public set this took the
  exponential group from 5/9 to 7/9 solved.
* **Iterative refinement and retries**: refinement stops at 1e-12 absolute plus 1e-12
  relative, which is about 15–25% faster on the 1-second LP/QP instances than 1e-13/1e-14.
  If a solve ends `unresolved`, the problem is solved again from scratch with tight
  refinement (1e-13/1e-14). SOC problems also get a third attempt with plain H-form SOC
  blocks. These attempts are deterministic, and a result is returned only after it passes
  the exact acceptance checks.
* **Equilibration**: Ruiz scaling of [P Aᵀ; A 0], uniform within non-separable cone
  blocks, and cost scaling as in Clarabel.
* **Initialization**: Clarabel-style KKT initialization with H = I when all cones are
  symmetric, and the central-point unit initialization when exponential cones are present.
* **Termination**: each iteration evaluates the official acceptance tests (relative
  primal/dual residuals, duality gap, infeasibility certificates with the matrix
  scalings a and h) on the *original* unscaled data, with a 0.9·1e-8 safety margin.
  It returns as soon as one passes. Result vectors are mapped back to the original
  coordinates; zero-cone slacks are set exactly to 0.

## Limitations
Outside the reduced path (for example PSD mixed with zero or SOC cones), PSD cones go
into the KKT system as dense blocks, and cones with more than 1300 svec entries are
reported `unresolved` immediately. Large SOC problems can still
stall in the last iterations.
