# Exponential propagation (FAMEXP2)

Computes `y = exp(-t*L) b0 + t*phi1(-t*L) b1` for a real symmetric PSD sparse `L`
given in full CSR, one request (one_shot) or 16 sequential requests (reuse).

## Algorithm

* **Two symmetric Lanczos processes**, one per right hand side, so each term can
  stop on its own. `b0 == 0` or `b1 == 0` skips a process entirely; `t == 0`
  returns `b0` unchanged.
* Symmetric three-term recurrence, **full re-orthogonalisation every step**
  (one classical Gram-Schmidt pass, second pass only when the correction is
  large).  This keeps the basis orthonormal to ~1e-15, which is what makes the
  stopping test below exact and prevents the ghost Ritz values that stiff
  problems (`t*||L|| ~ 1e4`) otherwise produce.
* Small projected problem: `eigh_tridiagonal` on `T_k`, giving
  `exp(-t*theta)` and `t*phi1(-t*theta) = t*(1-exp(-x))/x` from one
  eigendecomposition. Ritz values are clamped at 0 (L is PSD; negatives are
  round-off), so `exp` can never overflow.
* **Stopping test.** Because `V_k` is orthonormal, the exact distance between
  successive approximations is free in the small space:
  `||y_k - y_k'||_2 = beta * ||s_k - pad(s_k')||_2`. A geometric extrapolation
  of those differences estimates the remaining error, with a deliberately
  conservative factor (`max(rho/(1-rho), 4)`) and a 10x margin on
  `atol + rtol*||y||_inf`. Stiff operators with a near-null space converge
  non-monotonically (the dominant `phi1` mass sits at the smallest Ritz values,
  which appear late), so the margin is needed.
* `||y||_inf` (which sets the relative part of the accepted tolerance) is
  refreshed at doubling checkpoints `k = 2,4,8,...`; a final check re-runs any
  term whose error estimate exceeds the tolerance computed from the true
  `||y||_inf`.
* `n <= 1000` uses a dense symmetric eigendecomposition instead (exact, and
  cheap for the analytic control cases).
* Happy breakdown (`beta ~ 0`) means an invariant subspace: the answer is exact.

## Dependencies

Python 3 standard library, NumPy, SciPy (`scipy.sparse.csr_matrix`,
`scipy.linalg.eigh_tridiagonal`). Nothing is downloaded or compiled.

## Run / checks

    ./run.sh          # reads FAMEXP2 on stdin, writes READY then raw float64
    ./build.sh        # import validation only (no compilation needed)

Public check:

    python tools/run_local.py --manifest samples/manifest.json \
        --command "./run.sh" --repeats 3 --output dev_report.json

All six public sample cases (five synthetic families plus the analytic control)
were verified against an independent dense `eigh` reference in this workspace;
the Krylov error stayed 1e2-1e4 below the acceptance threshold on every case.

## Notes

* Memory: the two Krylov bases are capped at ~2.2 GB total
  (`mcap = min(1500, 2.2e9/(2*8*n))`), well inside the 4 GiB limit.
* stdout carries only `READY\n` plus 8n bytes per request; diagnostics go to
  stderr; the process exits 0 after the last vector.
