# FAMEXP2 exponential propagation: Chebyshev solver in C

`./run.sh` starts a compiled C program (`expo.c` + `cheb_body.h`). The program speaks
the FAMEXP2 wire protocol directly on fds 0/1. It computes, for every request,

    y = exp(-tL) b0 + t*phi1(-tL) b1        (L symmetric PSD, CSR input)

## Files

| file | purpose |
|---|---|
| `run.sh` | entry point; picks a prebuilt binary that passes `--check` on this CPU, else compiles `expo.c` with gcc |
| `expo.c` | I/O, setup, spectral bound, ordering, SELL-4 build, AVX2 kernels, driver |
| `cheb_body.h` | Chebyshev recurrence kernels and solver, instantiated for `double` and `long double` |
| `expo_v3`, `expo_v1` | prebuilt binaries (`-O3 -march=x86-64-v3` with AVX2/FMA kernels, and baseline x86-64) |
| `build_binaries.sh` | rebuilds both binaries from source (used during development; no `build.sh` is needed) |

Dependencies: Linux, the C library and libm only. No Python, NumPy or BLAS is used at run time.
gcc is needed only if neither prebuilt binary runs.

## Algorithm

1. **Input-independent setup, before READY (untimed).** A 64 MiB arena is mapped,
   advised for transparent huge pages and pre-faulted. Pipe buffers are enlarged to 1 MiB.
   The binary self-test (`--check`) runs from `run.sh` before the solver starts.
2. **Matrix read.** The raw CSR arrays are read straight into the arena. The
   Gershgorin bound `G = max_i sum_j |L_ij|` is computed for complete rows while the
   value array is still arriving.
3. **Spectral interval `[0, lh]`.** By default `lh = G`. When the expansion is long
   (estimated degree > 90 for one_shot, > 25 for reuse), a 12–80 step Lanczos run
   estimates `lambda_max`. It stops when the top Ritz value has converged, and then
   `lh = theta + max(2% theta, 1.5 * residual)`, capped at `G`.
4. **Chebyshev expansion.** On `[0, lh]`, `exp(-t lambda)` has coefficients
   `c_k = 2 (-1)^k e^{-beta} I_k(beta)` with `beta = t*lh/2`. These are computed
   with Miller's backward ratio recurrence. The coefficients `d_k` of
   `g(lambda) = (1 - e^{-t lambda})/lambda = t*phi1(-t*lambda)` (no removable singularity at 0)
   come from a stable backward recurrence that inverts multiplication by `(x+1)`.
   Tail sums `sum_{j>k} |c_j|`, `sum_{j>k} |d_j|` are tabulated.
5. **Forward three-term recurrence.** `v_{k+1} = 2 A~ v_k - v_{k-1}` with
   `A~ = (2/lh) L - I` is run on `b0` and `b1` together: either one vector, when one
   right-hand side is zero, or an interleaved pair. Each step is fused with the
   accumulation `y += c_k v0_k + d_k v1_k`. The loop stops at the first `k` with
   `3 * (tail_c(k) * V0 + tail_d(k) * V1) <= atol + 0.98 * rtol * ||y_k||_inf`, where `V` is
   the largest `||v_j||_inf` seen so far. The criterion is therefore adaptive to the
   requested precision and to the actual size of `y`.
6. **Safety nets.**
   * *Interval check.* With a correct interval, `||T_k(A~) b||_2 <= ||b||_2`. If any
     step exceeds this by `1e-6` relative, the Lanczos interval was too small, and the
     request is recomputed on the rigorous Gershgorin interval.
   * *Rounding check.* For large `t`, rounding errors in near-null modes are amplified
     by up to `max g = t`. After each solve, the bound
     `1.1e-16 * (deg+10) * (||b0||_inf + t ||b1||_inf) * max(1, G/lh)` is compared with
     `0.25 * (atol + rtol*||y||_inf)`. When the bound is larger, a *twin solve* runs
     first: the same polynomial at the same degree, with the right-hand sides scaled
     by pi/4. Scaling changes every rounding but leaves the mathematics and the
     truncation identical, so the difference between the two results measures the
     rounding error.
     If the two agree within `0.1 * tol`, their average is returned. Otherwise the request
     is recomputed with the same recurrence in x87 extended precision (`long double`).
     On the public data the twin check accepted cases with differences 100–10000× below
     the tolerance. It sent exactly the requests that need extended precision (log-spectrum
     operator, t = 1e4) to the redo. A wall-time guard keeps all redos inside the session budget.
   * Trivial sessions (`t = 0` or `L = 0`) return `b0 + t*b1` exactly.
7. **Fast kernels.** The sampled mean `|i-j|` of the nonzeros measures locality.
   SELL-4 is built when enough steps are expected: at least 8 for local orderings,
   at least 30 for random-access ones. If at least 150 steps are expected and the
   ordering is non-local, a BFS (Cuthill–McKee-like) order is tried. It is kept only
   if it halves the mean `|i-j|`, which happens for randomly numbered geometric
   graphs. Rows are sorted by length inside windows of 32. Chunks of 4 rows are
   stored column-major, so the gather and FMA chains of 4 rows run in parallel.
   On local orderings, AVX2 versions of the kernels are used (about 20% faster per
   step). On random-access graphs the scalar 4-row kernel is faster and is kept.
   Vectors are permuted on input and un-permuted on output. Cases with very few
   steps use the plain fused CSR kernel.

Reuse sessions share the setup (interval, coefficients, ordering, SELL arrays)
across all 16 requests. Each request is solved independently in its own precision mode.

## Public checks performed

These are local development checks; timings are from the shared development core.

* `tools/run_local.py` on `dev/manifest.json` with 3 repeats: 56/56 valid.
  The largest error is 0.32 of the tolerance.
* `tools/run_local.py` on `dev_full/manifest.json` with 3 repeats: 40/40 valid. The
  largest error is 0.29 of the tolerance, peak memory is 69 MB, and the total of median
  times is 3.1–3.3 s. The low-eta cases (n=60k–100k) take 6–35 ms each, mostly
  reading the input. The largest case (n=75k, nnz=820k, t*||L||=1e5, p10, degree
  about 1100) takes about 0.75 s.
* A randomized stress test against dense eigendecomposition: about 940 cases across
  development versions, 100 of them on the final binary. It covered random,
  disconnected, signed, potential-shifted, diagonal, path, log-spectrum (n up to 700)
  and random-geometric/random graphs (n of 3000–5000), with t*||L|| up to 1e5, all
  precisions, both scenarios, and dense, impulse, zero, tiny and "high-frequency"
  (`L*r`) right-hand sides. There were no failures. The worst error was 0.34 of the
  tolerance, which is the designed bound of the stopping rule; it is reached only for
  inputs dominated by the null space.
* Forced fallbacks: a Lanczos interval shrunk by 3%, 10% or 40% is detected by the
  growth monitor and recomputed correctly. Forced extended precision gives the same
  answers. `run.sh` still works with binaries that have no exec bit on a read-only
  directory (it copies them), and with no binaries at all (it compiles from source).
  `--check` fails on a deliberately broken AVX2 kernel.
* A full-scale synthetic reuse session (16 requests, nnz=820k, t*||L||=1e5, p10,
  mixed right-hand sides) takes about 12 s against the 240 s limit, at 86 MB peak memory.

Diagnostics go to stderr only when `EXPO_VERBOSE` is set. The other `EXPO_*` environment
variables are development and testing switches and are unset during evaluation.
