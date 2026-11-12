# Exponential propagation for sparse symmetric PSD operators (FAMEXP2)

Computes, for each request, `y = exp(-tL) b0 + t*phi1(-tL) b1`, where
`t*phi1(-tx) = (1 - exp(-tx))/x`.

## Files

* `run.sh`   – entry point (`python3 solve.py`, single-threaded BLAS env).
* `solve.py` – protocol I/O, spectral bounds, Chebyshev coefficients, driver.
* `cheb.c`   – C kernels: fused Chebyshev step, SpMV, bound pass, Lanczos steps.
* `build.sh` – rebuilds `libcheb_generic.so` (x86-64 baseline) and `libcheb_avx2.so`
  (AVX2/FMA, chosen at run time from `/proc/cpuinfo`). Prebuilt copies are shipped, and the
  build script always exits 0 and keeps those copies if compilation fails. If neither library
  loads, a pure NumPy/SciPy fallback path is used. Setting `FAMEXP_NOLIB=1` forces
  it, for testing.

Dependencies are Python 3, NumPy and SciPy (`scipy.fft.dct`, `scipy.special.ive`,
`scipy.linalg.eigh_tridiagonal`), plus gcc for the optional rebuild. Nothing is downloaded.

## Algorithm

1. **Input** is read straight into preallocated NumPy arrays with `readinto`, and the CSR
   arrays are used as delivered (uint64 indptr, uint32 indices). Nothing is converted or
   copied. Each result is written as raw float64 and flushed. The process ends with
   `os._exit(0)` to skip interpreter teardown.
2. **Spectral interval [lb, ub]**:
   * Rigorous: Gershgorin upper and lower bounds. When L is diagonally dominant (graph,
     signed or potential Laplacians), the tighter Anderson–Morley-type bound
     `max over stored off-diagonals (i,j) of L_ii + L_jj` also applies. It follows from
     `L = B B^T` and a scaled Gershgorin argument on `B^T B`.
   * Refined, only when the expansion is long: a short Lanczos run in C, without
     reorthogonalisation, using only the extreme Ritz values. It sets
     `ub = theta_max + 3 r_max` (r is the residual norm of the Ritz pair), capped by the
     rigorous bound. When `t * theta_min` is large it also sets `lb = theta_min - 3 r_min`.
     This removes the useless phi1 peak at 0 when there are no slow modes, which reduces
     both the number of terms and the rounding error.
   * Safety net: on a valid interval `|T_k(A) b|_inf <= |b|_2`. If the recurrence breaks
     this, the request is recomputed on the rigorous interval, which is then kept for the
     rest of the session.
3. **Chebyshev expansion** on [lb, ub]:
   * The exp coefficients are exact: `2(-1)^k e^{-t lb} ive(k, t(ub-lb)/2)`.
   * The phi1 coefficients come from a DCT-I of `-expm1(-tx)/x` on enough Chebyshev points.
     Coefficients below the DCT rounding floor are truncated.
   * Both series share one three-term recurrence, with `b0` and `b1` stored interleaved.
     A single fused C pass per degree does the SpMV for both vectors, the recurrence
     update, the accumulation into y, and the running max-norms.
4. **Adaptive stopping**: the iteration stops at the first degree k where
   `2*(S0_k*M0 + S1_k*M1) <= 0.05*(atol + rtol*|y_k|_inf)`. Here `S_k` is the tail sum of
   |coefficients| and `M` is the running maximum of `|T_j b|_inf`. The target error is
   therefore about 2.5% of the tolerance.
5. **Trivial cases**: when `t = 0` the result is `b0`. When L is zero, it is `b0 + t b1`. When
   one of the right-hand sides is identically zero, a single-vector kernel is used. Reuse
   sessions keep all setup (bounds, Lanczos, coefficients) across the 16 requests.

## Public checks performed

* All 6 public samples, through `tools/run_local.py` and `./run.sh`: valid. Against a dense
  `eigh` reference at p4/p7/p10, scaled errors are at most 0.08.
* 15 edge cases (n=1, nnz=0, stored zeros, t=0, constant diagonal, zero b0/b1/both,
  16-request reuse at p10, tiny t, diagonal potential, non-diagonally-dominant Gram PSD
  matrix, delta right-hand side): all pass, scaled errors at most 0.013.
* Full-scale stress tests (n≈120k, nnz≈1M random weighted graph and 2D mesh,
  `t*||L||_inf = 1e5`, p10) agree with a strict run on the rigorous interval within
  0.008 of the tolerance. One-shot wall time is about 1.2 s for the mesh and about 3 s for
  the graph.
