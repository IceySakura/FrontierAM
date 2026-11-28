# Exponential propagation (FAMEXP2) — submission

Computes, for a real symmetric positive semidefinite sparse `L`, a time `t`
and each pair of right hand sides `(b0, b1)`,

```
y = exp(-t L) b0 + t phi1(-t L) b1,      phi1(z) = (exp(z) - 1)/z,  phi1(0) = 1
```

## Layout

| file        | role |
|-------------|------|
| `run.sh`    | entry point executed by the evaluator (`exec ./solver`, Python fallback) |
| `build.sh`  | offline build: compiles `solver.c` into `./solver` |
| `solver.c`  | complete solver, C99 + libm, single process, single thread |
| `solve.py`  | pure Python/NumPy/SciPy implementation of the same algorithm (fallback only) |
| `sdk/wire.py` | copy of the provided protocol helper used by `solve.py` |

## Interface

`run.sh` writes `READY\n`, flushes, then speaks the binary FAMEXP2 protocol on
stdin/stdout exactly as described in `API.md`: header JSON, CSR arrays
(`indptr` uint64, `indices` uint32, `data` float64), then `requests` rounds of
`b0`,`b1` in / `y` out, and exits with code 0.  Diagnostics, if any, go to
stderr; stdout carries only protocol bytes.  Nothing is cached across sessions
(every session is a fresh process).

## Algorithm

Both matrix functions are evaluated together by one Chebyshev expansion over a
spectral interval that provably contains the spectrum of `-t L`.

1. **Spectral interval.** `Lam = lambda_max(L)` is bounded from above without
   any eigen-solve: `rho(L) = rho(D^-1 L D) <= ||D^-1 L D||_inf` for every
   positive diagonal `D`.  Four cheap passes give `D = I`, `D = diag(rowsum)`,
   `D = diag(sqrt(rowsum))`, `D = diag(sqrt(|diag L|))` and the smallest bound
   is used (within a few percent of `lambda_max` on all public families, which
   matters because the Chebyshev degree grows like `sqrt(t*Lam)`).  The
   interval in `z = -t*lambda` is `[-t*Lam, 1e-6*t*Lam]`; the tiny positive
   right end keeps round-off level negative eigenvalues of the *stored* matrix
   inside the interval.  `L` PSD gives the lower end.

2. **Coefficients.** With `z = c + r u`, `u in [-1,1]`:
   * `exp(z) = sum_k alpha_k T_k(u)`, `alpha_k = (2 - [k=0]) I_k(r) e^{-r} e^{c+r}`,
     evaluated through the scaled Bessel values `I_k(r)e^{-r}` from a downward
     Miller recurrence normalised by `I_0 + 2 sum_{k>=1} I_k = e^r` (no overflow,
     ~1e-15 relative accuracy).
   * `t phi1(z) = sum_k beta_k T_k(u)`: multiplying the series by `z = c + r u`
     must reproduce `exp(z) - 1`, which gives the three term relation
     `G_{j-1} = 2(E_j - c G_j)/r - G_{j+1}` (with `E = alpha`, `beta = t*G`) plus
     two closing equations for `G_0, G_1`.  Run downward from `k = M+1` it is
     stable (the homogeneous solutions are bounded) and matches a long double
     DCT of `t phi1` to ~1e-14 relative.

3. **Operator series.** The forward three term recurrence
   `v_{k+1} = 2 U v_k - v_{k-1}`, `U = (A - c I)/r`, `A = -t L`, is summed
   directly into the output by one fused sparse kernel per degree that carries
   both right hand sides at once: the matrix is streamed once per step and
   reused for the two columns, and the two columns are stored *interleaved*
   (`x[2i], x[2i+1]`) so a single cache line answers the gather for both right
   hand sides.  Measured 1.1-1.2x faster than separate column arrays on
   graph-like sparsity, neutral elsewhere.  When one right hand side is zero the
   kernel switches to a single column variant (~40% cheaper per step).  Cost per
   step is one sparse matvec plus a handful of vector passes; no Krylov basis is
   stored, so memory stays O(n + nnz) (~70 MB at n = 1.2e5, nnz = 1e6).

4. **Adaptive degree with a rigorous stopping test.** Because `|T_k| <= 1` on
   the interval, truncating after `m` terms costs at most
   `(sum_{k>m}|alpha_k|) * max_j ||v_j||_inf + (sum_{k>m}|beta_k|) * max_j ||w_j||_inf`,
   and both tail sums are known exactly from the coefficients.  The loop tracks
   `max|v_k|` and `||S||_inf` for free inside the kernel and stops as soon as
   `4 * bound <= atol + rtol * ||S||_inf`, so easy cases (small `t*Lam`, or a
   large output that makes the mixed tolerance loose) stop early and only
   genuinely stiff/high accuracy cases pay for a long series.  The degree cap is
   `M ~ sqrt(t*Lam*(log(1/atol)+10)) + 16`.

5. **Exact small-component path, chosen by cost model.** If every connected
   component of the graph of `L` has at most 24 nodes (block diagonal / locally
   diagonal operators, and any tiny operator), the solver can precompute
   `exp(-t B)` and `t*phi1(-t B)` per block once per session with a cyclic Jacobi
   eigensolve and apply them per request in O(n*k) — exact rather than truncated.
   Building the blocks costs two transcendentals per eigenvalue plus the
   eigensolves, which is only worth it when the series would need enough degrees
   to pay for it, so after component detection a calibrated nanosecond cost model
   (`40n + 15*sum k^3 + 7*sum k^2*requests + 5(n+nnz)` versus
   `1.4 * M_est * requests * 2.5 * (nnz+n)`) picks the cheaper path; borderline
   cases stay on the series.  Component search aborts as soon as a component
   exceeds 24 nodes, so detection on a connected operator costs microseconds.
   Measured on a 2x2-block log-spectrum operator with n = 1.2e5: the model picks
   the series at `t*||L||_inf <= 1` (14.5 ms vs 19.7 ms) and the blocks above
   that (23 ms vs 234 ms at 1e4, 23 ms vs 714 ms at 1e5; 61 ms vs 9452 ms for a
   16-request session at 1e5).  The public log-spectrum sample runs exactly, at
   0.9 ms instead of 5 ms.

6. **Degenerate inputs.** `t = 0` returns `b0`; `L = 0` (or `nnz = 0`) returns
   `b0 + t*b1`; a zero right hand side is dropped so a one column kernel runs
   (about 40% cheaper per step than the two column kernel).  Singularity of `L`
   needs no special handling because `phi1` is entire and `phi1(0) = 1`.

A wall clock guard stops the series at 85% of the session budget so a pathological
input degrades in accuracy rather than missing the limit.  The process finishes
with `_exit(0)`: all protocol bytes were already written by unbuffered `write()`
syscalls, so skipping libc teardown is worth ~0.1 ms on short sessions and
~1 ms on large ones.

## Dependencies

Build: `gcc` (any C99 compiler) and `libm`.  `build.sh` compiles to a temporary
name and tries `-O3 -march=x86-64-v3 -mtune=native`, then `-O3 -march=x86-64-v2`,
then plain `-O3`, then `-O2`; each candidate must pass a smoke run (empty stdin
must make it exit 1 on a short read) before it replaces `./solver`, so a failed
build never destroys a working binary and `build.sh` itself never fails the
submission.  Runtime: nothing but libc/libm.  `run.sh` re-checks the binary the
same way before `exec`ing it and falls back to `python3 solve.py` if it is
missing or unusable; that fallback needs only the preinstalled `numpy`/`scipy`
and reproduces the same algorithm (~10x slower, all public cases pass with it).

## Public checks performed

```sh
cd /workspace/task
python3 tools/run_local.py --manifest samples/manifest.json \
        --command "/workspace/submission/run.sh" --repeats 3 --output dev_report.json
```

All six public cases (five synthetic structure families plus the analytic
control) pass with three repeats; the worst scaled error against the published
`y_ref` is ~0.13 of the allowed tolerance (the log-spectrum case is now handled
exactly by the small-component path, 0.003).  The Python fallback produces
bit-comparable scaled errors, which cross-checks the C kernel, the Bessel
recurrence and the `phi1` coefficient recurrence against an independent
implementation.  Private regression suites that were run:

* 26 cases, n = 1200-1444, six families, `t*||L||_inf` from 1e-2 to 1e5,
  p4/p7/p10, checked against dense `eigh`: all pass, worst error 0.24 of
  tolerance.
* 18 protocol/edge cases checked against dense `eigh`: n = 1, zero matrix,
  `nnz = 0`, empty rows, `t = 0`, zero `b0`, zero `b1`, both zero, singular
  Laplacian with `t = 500`, stiff operator with a spectral gap, 2x2 block
  diagonal at `t = 5000`, and 16-request reuse sessions (moderate and stiff).
  All pass, worst error 0.16 of tolerance.
* Large cases (n = 1.0e5-1.2e5, nnz = 6e5-8.6e5, `t*||L||_inf` = 1e2 and 1e5,
  p7/p10) checked against an independent long double Chebyshev expansion:
  worst error 0.08 of tolerance, including the cancellation dominated case
  where `||y||_inf = 1.3e-3` and `atol` sets the tolerance.
* Worst measured one_shot session on the largest stiff case: 3.2 s (limit 15 s).
  Worst measured reuse session (16 requests, `t*||L||_inf` = 1e5, p10,
  nnz = 8.6e5): 50 s (limit 150 s).  Peak RSS ~40 MB (limit 4 GiB).
