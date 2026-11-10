# Public calibration protocol (declared before the first run)

`chebyshev2` has no case-family switch and no timing-fitted numerical constant.
Before running any public validation case, all fixed constants and decision rules
were declared analytically as follows:

## 1. Spectral interval bound (`lanczos`)

1. Compute the PSD Gershgorin upper bound in `long double`:
   `lambda_gersh = max_i (L_ii + sum_{j != i} |L_ij|)`, rounded upward with
   `nextafter` and inflated by `1 + (8 + max_row_nnz) * DBL_EPSILON`, preserving
   exact zero when `L = 0`.
2. During the same pass, compute the exact 2-norm lower bound
   `lambda_lower = max_i max(L_ii, sqrt(sum_j L_ij^2))`.
3. Run a short full-reorthogonalization symmetric Lanczos process on `L` only
   when `t > 0`, `n >= 64`, `lambda_lower < 0.92 * lambda_gersh`, and
   `requests * sqrt(a_gersh) >= 8`, where `a_gersh = t * lambda_gersh / 2`.
4. The deterministic starting vector combines the row-sum-weighted SplitMix64
   sign vector in original vertex numbering with a localized spike aligned with
   the maximum-Gershgorin row, normalized in 2-norm. This makes the starting
   vector and resulting bound invariant to vertex permutations and independent
   of every right-hand side.
5. Lanczos runs for at most `m_max = 24` steps, stopping early at step
   `m >= 10` if the largest Ritz pair `(theta_m, u_m)` of `T_m` satisfies
   `rho_m = beta_m * |u_{m,m}| <= 1e-3 * theta_m` and
   `theta_m - theta_{m-3} <= 1e-3 * theta_m`.
6. The Lanczos bound is accepted only if `theta_m >= lambda_lower`,
   `rho_m <= 0.05 * theta_m`, and `theta_m - theta_{m-3} <= 0.02 * theta_m`.
   When accepted,
   `lambda_upper = min(lambda_gersh, 1.05 * (theta_m + rho_m + 0.25 * beta_m))`.
   Otherwise `lambda_upper = lambda_gersh`.
7. During the forward Chebyshev recurrence, if any iterate norm exceeds
   `2 * ||b||_2` (indicating an eigenvalue outside `[-1, 1]`), the solver
   immediately falls back to `lambda_gersh`, rebuilds coefficients, and
   recomputes the request.

## 2. Chebyshev coefficient construction

1. With `a = t * lambda_upper / 2`, generate scaled modified-Bessel values
   `E_k = exp(-a) I_k(a)` in `long double` via Miller backward ratio recurrence
   from `top = max(160, ceil(12 * sqrt(a + 1) + 128))` and normalize with
   `E_0 + 2 * sum_{k>=1} E_k = 1`.
2. Form `f_0 = E_0`, `f_k = 2 * (-1)^k * E_k` for `k >= 1`, and obtain the
   coefficients `q_k` of `g(x) = t * phi_1(-t*x)` from the three-term
   Chebyshev identity `(1 + z) q(z) = t * (1 - f(z)) / a` in `long double`.
3. Precompute cumulative absolute tail sums
   `f_tail[k] = sum_{j>=k} |f_j|` and `q_tail[k] = sum_{j>=k} |q_j|` in
   `long double`.

## 3. Fused three-term recurrence (`fused`) and adaptive truncation (`adaptive`)

1. Using the affine operator `Z = (2 / lambda_upper) * L - I`, propagate
   `v_k^(0) = T_k(Z) b0` and `v_k^(1) = T_k(Z) b1` forward via
   `v_0 = b`, `v_1 = Z b`, `v_{k+1} = 2 Z v_k - v_{k-1}`, accumulating
   `y_k = sum_{j=0}^k (f_j v_j^(0) + q_j v_j^(1))` in a single matrix pass per
   degree `k`. When `b0 = 0` or `b1 = 0`, only the active single vector is
   propagated; when both are nonzero, `(v_k^(0), v_k^(1))` are propagated as a
   16-byte vertex-interleaved pair in one SpMV pass.
2. Declared safety factor: `S = 8` (identical to `panel/chebyshev`).
3. At degree `k >= 2` in the monotone Bessel decay regime
   `r_k = I_{k+1}(a) / I_k(a) <= 0.9`, form the 4-step iterate infinity-norm
   envelopes capped by the 2-norm bound:
   `W_k^(0) = min(||b0||_2, 2 * max_{k-3 <= m <= k} ||v_m^(0)||_inf)` and
   `W_k^(1) = min(||b1||_2, 2 * max_{k-3 <= m <= k} ||v_m^(1)||_inf)`.
4. Combine the coefficient-tail bound and the step-norm geometric bound:
   `E_k = max(f_tail[k+1] * W_k^(0) + q_tail[k+1] * W_k^(1), (r_k / (1 - r_k)) * max(||Delta y_k||_inf, ||Delta y_{k-1}||_inf))`.
5. Stop when for two consecutive degrees,
   `E_k <= (atol + rtol * max(||y_k||_inf - E_k, 0)) / S`.

## 4. Sparse matrix format (`sell`) and bandwidth ordering (`rcm`)

1. Measure the average row-index span
   `avg_jump = (1 / nnz) * sum_{i, j in row(i)} |i - j|`.
2. When `n >= 4096`, `avg_jump > 0.04 * n`, and
   `requests * sqrt(a_gersh) >= 25`, compute a symmetric Reverse Cuthill–McKee
   ordering (with a 2-sweep pseudo-peripheral root finder per component) and
   retain it only if it reduces `avg_jump` by at least 30%
   (`avg_jump_rcm < 0.70 * avg_jump`).
3. Convert the matrix into SIMD SELL-`C`-`sigma` format with slice height
   `C = 4` (256-bit AVX2 width for `binary64`) and sorting window `sigma = 64`,
   combining the RCM and local `sigma`-window permutations into a single
   symmetric permutation `P` applied once per request to `(b0, b1)` and `y`.
4. Reuse `lambda_upper`, the Chebyshev coefficients, the permutation, and the
   SELL-`4`-`64` structure across all requests of a session without retaining
   any RHS-dependent state.

## Correction 1 — 2026-11-10

Public validation of the plain binary64 forward recurrence revealed a
uniform rounding defect, not a truncation defect: on
`syn_geometric_b_p10_k16` (`a = 50000`, degree at the coefficient cap) two
strongly damped impulse requests had scaled errors 0.27 and 0.38 against the
0.1 acceptance margin, the error was unchanged when the safety factor was
raised (truncation was already negligible), and the backward-Clenshaw panel
member reaches about 0.007 on the same case. The forward three-term
recurrence amplifies per-step SpMV and state rounding roughly linearly in
the degree, which the declared factor `S = 8` does not cover at large `a`.

The correction is uniform and analytic; no constant was fitted and no
timing was used to select it:

1. the recurrence state `(v_{k-1}, v_k)` is carried in `long double`;
2. each SpMV is applied to the binary64 hi/lo split of the state and the
   two results are recombined in `long double`, so the operator sees the
   state to well below binary64 rounding;
3. every SpMV accumulation (CSR, SELL scalar, and SELL AVX2/FMA) uses
   Kahan–Babuška (Neumaier) compensated summation with FMA two-product
   error terms, and returns its accumulated error term, which is included
   in the `long double` recombination (a plain kernel suffices for the
   lo part, whose rounding is second-order);
4. the output accumulation `y += f_k v_k^(0) + q_k v_k^(1)` is likewise
   Kahan-compensated.

The compensated path is engaged only when the worst-case a-priori degree
(`choose_degree` at output scale 0) exceeds 512, the degree regime in
which forward-recurrence rounding amplification (growing with the degree)
can reach the `1/8` budget; at smaller degrees the plain binary64 fused
recurrence with Kahan output accumulation is retained. The threshold is
a uniform analytic degree bound, not a case switch.

The safety factor `S = 8`, all thresholds in Sections 1–4, and the
adaptive stopping rule are unchanged. After the correction the same case
has maximum scaled error 0.002 (Lanczos disabled) / 0.080 or below in
every configuration, and all 72 public cases pass in the default and in
each single-technique-disabled configuration (see `VALIDATION.md`).

## Correction 2 — 2026-11-10

Public validation with `CHEB2_DISABLE=adaptive` revealed that the
a-priori `||b||_2` tail rule at the declared factor `S = 8` does not by
itself guarantee the 0.1 acceptance margin: on `control_diagonal`
(`L = diag(4, 0)`, spectrum at the Chebyshev endpoints `z = ±1`, where
`|T_k(z)| = 1` and the coefficient-tail bound is attained rather than
loose) it gave scaled error 0.102. The bound `error <= (atol +
rtol*scale)/8` only ever promised about 1/8 of the acceptance threshold.

The correction is uniform and analytic; no constant was fitted by timing
and no case switch was added: the a-priori degree rule used when
`adaptive` is disabled (both its initial degree and its post-hoc
output-relative degree check) uses safety factor 64 in place of 8, so
the attained-bound worst case is at most about 1/64 of the acceptance
threshold. The adaptive rule (Correction 1 path) keeps `S = 8`, since
its certified gate is backed by the measured iterate envelope. After
the correction `control_diagonal` with `adaptive` disabled has scaled
error 0.0042.
