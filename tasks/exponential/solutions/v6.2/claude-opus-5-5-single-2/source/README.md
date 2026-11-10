# Exponential propagation solver (FAMEXP2)

Computes, for a symmetric positive semidefinite sparse `L` (CSR), time `t` and
one or 16 sequential right-hand-side pairs,

    y = exp(-tL) b0 + t*phi1(-tL) b1

over the FAMEXP2 stdin/stdout protocol (see the task's `API.md`).

## Files

| file | purpose |
|---|---|
| `expprop.c` | complete solver source (C99 + GCC vector extensions; needs only libc/libm) |
| `run.sh` | entry point: execs `bin/expprop_avx2` when `/proc/cpuinfo` reports AVX2 and FMA, otherwise `bin/expprop_generic` |
| `build.sh` | optional rebuild of both binaries from source: `gcc -O3 -mavx2 -mfma expprop.c -lm` and `gcc -O3 expprop.c -lm`. A binary is replaced only if its compile succeeds; the script always exits 0 |
| `bin/` | prebuilt x86-64 Linux binaries, built in the task image with exactly those commands |

Dependencies: gcc 12 and glibc/libm from the provided image. Nothing else is needed at run time (no Python, BLAS or LAPACK).

## Protocol handling

Before `READY`, the program does only input-independent work. It maps a 112 MiB
arena (with a transparent-huge-page hint) and pre-faults it, so the timed session
does not take page faults for its working set. After `READY` it:

* reads the magic, JSON header and CSR arrays with raw `read(2)` into their final buffers,
  after enlarging the pipe buffer to 1 MiB;
* for each request, reads `b0` and `b1`, then writes exactly `n` float64 values with
  `write(2)` (unbuffered, so nothing waits on a flush);
* ends with `_exit(0)` right after the last result.

Diagnostics go to stderr only, a few lines per request.

## Algorithm

1. **Spectral interval.** Gershgorin gives a guaranteed `[lo, hiG]`, with `lo` clamped
   to at least 0 since L is PSD. When the expected work is large (estimated degree x
   requests > 350), the top of the interval is tightened by 50 Lanczos steps from a fixed
   pseudo-random vector: `hi = min(hiG, 1.004*(theta_max + 2*r))`, where `r` is the residual
   norm of the top Ritz pair. If the Chebyshev vectors grow by more than 1e4 (only possible
   when `hi` is below `lambda_max`), the solver falls back to `hiG` and restarts. This
   recovery path was tested with a deliberately underestimated `hi`.
2. **Chebyshev expansion.** `f0(x) = exp(-t x)` and `f1(x) = (1 - exp(-t x))/x` (`= t` at 0)
   are expanded on `[lo, hi]`. The coefficients come from a DCT at Chebyshev nodes using an
   exact cosine table. Nodes are formed as `lo + 2h sin^2(.)`, which avoids cancellation
   near `x = 0`; with `c + h cos(.)` the node rounding left a 1e-15 coefficient noise floor.
3. **Fused recurrence.** Both right-hand sides go through one three-term recurrence. One
   sweep over the matrix computes `T_{k+1}(X) [b0 b1]` (interleaved pairs, a 2-column SpMM),
   adds `a0_k T_k b0 + a1_k T_k b1` to `y`, and tracks infinity norms. Additional measures:
   * For multi-step solves, the matrix is stored once per symmetric pair (upper triangle
     plus a scatter accumulator), but only if the input is exactly symmetric.
   * Graphs with a deep BFS level structure and poor locality are reordered by reverse
     Cuthill-McKee. The permutation is applied to the inputs and undone on output.
   * Unweighted graphs (all off-diagonal values equal) use an index-only kernel.
4. **Truncation control (adaptive degree).** After step `k` the truncation error is bounded
   by `E = 2*(S0_k*M0 + S1_k*M1)`, where `S_k` is the coefficient tail sum and `M` is the
   largest observed infinity norm of the Chebyshev vectors. A request stops when
   `E <= 0.05*(atol + rtol*||y_k||_inf)`.
5. **Rounding control.** For large `t`, `f1` is large near `x = 0`, and the three-term
   recurrence can amplify rounding beyond the tolerance. One case in the public set
   failed at 1.7x the threshold without this step. Handling:
   * An indicator `eps * sum_k k*|a_k| * M` is evaluated after each request.
   * If it exceeds 1% of the tolerance, the request is recomputed in double on a slightly
     wider interval, which gives independent rounding.
   * If the two results agree to within 5% of the tolerance, the first is accepted.
   * Otherwise the request is recomputed with an x87 extended-precision (`long double`)
     recurrence on the unscaled matrix.
   * A time guard skips the extended-precision step if it would endanger the session's
     wall-clock limit. The average of the two double results is returned instead.
6. **Special cases.** `t = 0` returns `b0`. A scalar operator (`L = cI`, including `L = 0`)
   is evaluated in closed form. SSE flush-to-zero/denormals-are-zero is enabled, so
   subnormal intermediates (`exp(-t*lo)` near the underflow limit) cannot slow the
   kernels down. This only affects values below 1e-307, far under any tolerance.

## Public checks performed

* **Local harness.** `tools/run_local.py` with 3 repeats on all of `dev/` (56 sessions,
  including the 8 controls) and `dev_full/` (40 sessions): all pass. The largest scaled
  error (error / threshold) is 0.031. The generic (non-AVX2) binary passes the same set.
* **Rounding check.** The double path alone fails `dev/syn_logspectrum_p10_k16` (1.7x the
  threshold); with the second-opinion and long-double steps the worst is 0.03.
* **Synthetic edge cases.** All pass, worst 0.024. They were checked against dense
  eigendecomposition or SciPy `expm_multiply` on the augmented system:
  * isolated nodes and several components at `t*||L|| = 1e4`;
  * a pure diagonal with log-spread entries;
  * a nonsingular operator with tiny output (p4 and p10);
  * values that are asymmetric at the 1e-16 level (falls back to full storage);
  * an SPD matrix with positive off-diagonals;
  * a 100k-node graph with a 40k-degree hub.
* **Timing.** Measured on the development machine, which is noisy:
  * small `t*||L||` dev_full sessions take 3-25 ms;
  * `syn_geometric_b_p10` (n=75k, nnz=821k, `t*||L|| = 1e5`) takes about 1.2 s;
  * a synthetic 16-request version of it takes 17 s, or 82 s with extended precision forced
    on every request, against the 240 s limit;
  * peak RSS is under 200 MB.

A Lanczos (Krylov) variant was prototyped and rejected. On network-like spectra it needed
fewer matrix-vector products, but its difference-based stopping rule accepted a stagnated
iterate (`dev/syn_signed_p10_k16`, request 3: every eigenvalue is above about 0.19 and the
true result is tiny). The Chebyshev tail bound has no such failure mode.
