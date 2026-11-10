# Conic QP solver (HSDE interior point, C++)

## Usage

```sh
./build.sh                      # optional; compiles bin/conic_solver (run.sh also compiles on demand)
./run.sh --input IN.npz --output OUT.npz
```

`run.sh` `exec`s the native binary `bin/conic_solver`. No Python is used at solve time
(importing numpy/scipy alone costs 0.25–0.7 s, too much for the 1 s limits).

## Dependencies (fixed versions, all preinstalled in the image)

* g++ 12.2.0 (C++17), built with `-O3`
* zlib 1.2.13 runtime (`libz.so.1`, linked as `-l:libz.so.1`) for inflating the
  deflate-compressed NPZ input. The zlib API is declared in-source (no zlib headers
  are installed). Output NPZ files are written as stored (uncompressed) zip archives.

No other libraries (no BLAS/LAPACK, no SuiteSparse) are used.

## Method

Source: `src/io.hpp` (NPZ/zip/.npy reader+writer, minimal JSON), `src/linalg.hpp`
(sparse CSC utilities, AMD ordering, LDLᵀ), `src/solver.cpp` (IPM).

* **Algorithm**: primal-dual interior point method on the homogeneous self-dual
  embedding for quadratic objectives, following Clarabel (Goulart & Chen,
  *Clarabel: an interior-point solver for conic programs with quadratic objectives*, 2024):
  residual system `Px + A'z + qτ = 0`, `Ax + s = bτ`, `q'x + b'z + x'Px/τ + κ = 0`,
  Mehrotra predictor–corrector with σ = (1−α_aff)³, step fraction 0.99.
  Each iteration factors one quasi-definite KKT matrix `[P+εI, A'; A, −(H+εI)]`
  and performs three solves (constant `[-q; b]` system, affine, combined); `dτ`
  is recovered from the scalar Schur complement as in Clarabel.
* **Cones**: Nesterov–Todd scaling for nonnegative, second-order (hyperbolic
  rotation form as in ECOS, Domahidi et al. 2013) and PSD cones (Cholesky + SVD
  construction, Todd–Toh–Tütüncü); the exponential cone uses a primal-barrier
  scaling `H = (μ∇²f(s))⁻¹` with backtracking to keep iterates interior.
* **Linear algebra**: own implementation of approximate minimum degree ordering
  (Amestoy, Davis & Duff; structure follows CSparse `cs_amd`), elimination-tree
  up-looking LDLᵀ (QDLDL-style) with static (1e-8) and dynamic sign-based
  regularization, and iterative refinement against the unregularized KKT matrix.
* **Scaling**: Ruiz equilibration of `[P A'; A 0]` (cone-block uniform for
  SOC/exp/PSD rows) plus cost scaling.
* **Termination**: the checker's exact acceptance conditions are evaluated every
  iteration on the unscaled candidate `(x/τ, s/τ, z/τ)` in original coordinates
  (relative primal/dual residuals, absolute-or-relative gap, cone defects).
  Infeasibility certificates are tested with the checker's normalized
  certificate conditions (for dual infeasibility both the iterate `s` and
  `s = −Ax` are tried). The best candidate found is always written.

## Experimental basis

Developed against the 163 public instances with `public-dev/io/check_result.py`
(development core shared with other jobs, so timings are approximate).

* Interface examples: LP/QP, SOC, PSD and both infeasibility certificates verify.
* Linear/quadratic: the large majority of the 1-second instances (up to
  n≈16k, m≈51k) verify in 0.01–0.6 s, including primal-infeasible ones.
  Known failures: a few degenerate LPs/QPs where the dual residual stalls at
  1e-6–1e-3; instances with dense Cholesky fill (e.g. n=22k LP, nnz(L)=7M) are too
  slow with the non-supernodal LDLᵀ; the multi-million-variable QPs are solved
  but 1.5–2× slower than their limits (and the largest exceed 4 GiB in development).
* Second-order cone: iterates reach ~1e-7–1e-8 accuracy and then the NT-scaled
  KKT solves lose accuracy (non-finite directions near the cone boundary);
  most SOC dev instances therefore end slightly above the 1e-8 tolerance.
  Fixes tried at the end: analytic det(λ), refinement that only accepts improving
  corrections, σ lower bound — not sufficient.
* Exponential cone: converges linearly to ~1e-7 on the example, not 1e-8.

Iterative-refinement fix (reject non-improving corrections) was checked for
regressions on a 20-instance LP/QP sample (18/20 verified, unchanged).

Environment variables (`CONIC_VERBOSE`, `CONIC_TIME_CAP`, ...) are for development
only; defaults are used in evaluation.
