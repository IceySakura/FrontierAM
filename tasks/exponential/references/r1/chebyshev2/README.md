# Chebyshev literature implementation v2 (`chebyshev2`)

This directory contains `chebyshev2`, a second independent C99 human-baseline
implementation in the Chebyshev polynomial family for the FAMEXP2
exponential-propagation task:

`y = exp(-t*L) b0 + t * phi_1(-t*L) b1`, where
`g(x) = t * phi_1(-t*x) = (1 - exp(-t*x)) / x` and `g(0) = t`.

Like `panel/chebyshev`, it expands `exp(-t*L)` and `t*phi_1(-t*L)` directly on
the spectral interval `[0, lambda_upper]` using scaled modified Bessel
coefficients `exp(-a) I_k(a)` (`a = t * lambda_upper / 2`) and the three-term
Chebyshev coefficient recurrence for `g(x)`, without forming any matrix inverse
or subtracting `b1 - L*b0`. It strengthens `panel/chebyshev` with five
published numerical and high-performance sparse linear algebra techniques:

1. **Short Lanczos spectral upper bound with rigorous fallback**:
   Before constructing Chebyshev coefficients, a short symmetric Lanczos run
   (`m <= 24` with full reorthogonalization) estimates the largest eigenvalue
   of `L` from a deterministic, permutation-invariant starting vector aligned
   with the row-sum distribution. The bound combines the largest Ritz value,
   Krylov residual, and a multiplicative safety factor, is capped above by the
   PSD Gershgorin bound, and falls back to Gershgorin whenever the residual or
   step movement is not yet small or when the problem is already low-degree. A
   runtime iterate-norm guard during the Chebyshev recurrence also falls back
   to Gershgorin if any iterate ever exceeds the `[-1, 1]` Chebyshev envelope.
2. **Single fused three-term recurrence**:
   Instead of running two separate backward Clenshaw recurrences (and
   recomputing from scratch when the output is strongly damped), a single
   forward three-term Chebyshev recurrence propagates `(T_k(Z) b0, T_k(Z) b1)`
   together as a 16-byte vertex-interleaved pair (or a single vector when one
   right-hand side is zero) and accumulates `y` directly in the SpMV slice
   epilogue. Each polynomial degree reads the sparse matrix once.
3. **Certified adaptive a-posteriori truncation**:
   In the monotone Bessel decay regime (`I_{k+1}(a) / I_k(a) <= 0.9`), the
   omitted infinite tail is bounded using both the precomputed `long double`
   coefficient tails multiplied by the local 4-step iterate infinity-norm
   envelope (capped by `||b||_2`) and the geometric series bound of consecutive
   step updates, tested against the certified output-relative tolerance
   `(atol + rtol * max(||y_k||_inf - E_k, 0)) / 8`.
4. **SELL-`C`-`sigma` sparse format and Reverse Cuthill–McKee (RCM) ordering**:
   Poorly ordered graphs (`avg_jump > 0.04 * n`) with sufficient session work
   are symmetrically reordered with Reverse Cuthill–McKee when RCM reduces
   average index span by at least 30%. The matrix is stored in SIMD-friendly
   SELL-`4`-`64` format (`C = 4` matching 256-bit AVX2 `binary64` vectors,
   sorting window `sigma = 64` using stable degree sorting) and evaluated with
   AVX2/FMA kernels.
5. **Session-wide reuse**:
   The RCM/SELL permutation, SELL-`4`-`64` matrix structure, Lanczos/Gershgorin
   spectral bound, and Chebyshev coefficient tables depend only on `(L, t)` and
   are constructed once per session and reused across all 16 requests of a
   reuse session without retaining any right-hand-side state.

## Runtime ablation switch

Every technique can be disabled individually or in combination at run time via
the comma-separated environment variable `CHEB2_DISABLE`:

- `CHEB2_DISABLE=lanczos` — use the PSD Gershgorin bound directly;
- `CHEB2_DISABLE=fused` — run two separate recurrences for `b0` and `b1`;
- `CHEB2_DISABLE=adaptive` — use the a-priori `||b||_2` tail degree rule with
  post-hoc output-relative degree check;
- `CHEB2_DISABLE=sell` — use CSR SpMV kernels instead of SELL-`4`-`64`;
- `CHEB2_DISABLE=rcm` — disable Reverse Cuthill–McKee reordering;
- `CHEB2_DISABLE=reuse` — rebuild matrix format, bound, and coefficients per
  request.

By default (`CHEB2_DISABLE` unset or empty), all five techniques are enabled.

## Literature

- H. Tal-Ezer and R. Kosloff, “An accurate and efficient scheme for
  propagating the time dependent Schrödinger equation,” *J. Chem. Phys.*
  **81** (1984), 3967–3971.
- L. Bergamaschi and M. Vianello, “Efficient computation of the exponential
  operator for large, sparse, symmetric matrices,” *Numerical Linear Algebra
  with Applications* **7** (2000), 27–45.
- L. Bergamaschi, M. Caliari, and M. Vianello, “Efficient approximations of
  the exponential operator for discrete 2D advection-diffusion problems,”
  *Numerical Linear Algebra with Applications* **10** (2003), 271–289; and
  M. Caliari, M. Vianello, and L. Bergamaschi, “Interpolating discrete
  advection-diffusion propagators at Leja sequences,” *J. Comput. Appl. Math.*
  **172** (2004), 79–99.
- Y. Saad, *Numerical Methods for Large Eigenvalue Problems*, 2nd ed., SIAM,
  2011.
- Y. Zhou, Y. Saad, M. L. Tiago, and J. R. Chelikowsky, “Self-consistent-field
  calculations using Chebyshev-filtered subspace iteration,” *J. Comput. Phys.*
  **219** (2006), 172–184 (and Y. Zhou and R.-C. Li, “Bounding the spectrum of
  large Hermitian matrices,” *Linear Algebra Appl.* **435** (2011), 480–493).
- M. Kreutzer, G. Hager, G. Wellein, H. Fehske, and A. R. Bishop, “A unified
  sparse matrix data format for efficient general sparse matrix-vector
  multiplication on modern processors with wide SIMD units,” *SIAM J. Sci.
  Comput.* **36**(5) (2014), C401–C423.
- E. Cuthill and J. McKee, “Reducing the bandwidth of sparse symmetric
  matrices,” *Proc. 1969 24th National Conference*, ACM (1969), 157–172.
- N. J. Higham, *Functions of Matrices: Theory and Computation*, SIAM, 2008.

## Build and run

`bash build.sh` compiles both `chebyshev2_avx2` (`-march=x86-64-v3`) and
`chebyshev2_generic` (`-march=x86-64`). `run.sh` dispatches to
`chebyshev2_avx2` when AVX2 and FMA are supported by the CPU.

```sh
bash build.sh
./run.sh
```
