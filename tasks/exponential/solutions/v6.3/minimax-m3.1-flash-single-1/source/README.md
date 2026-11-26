# FAMEXP2 — exponential propagation for sparse symmetric PSD operators

## Task

For a fixed real symmetric positive semidefinite sparse matrix `L`, a time `t`
and an accuracy pair `(rtol, atol)`, a session receives one (`one_shot`) or
sixteen (`reuse`) right-hand-side pairs `(b0_j, b1_j)` and must return, for
every request,

```
y_j = exp(-t L) b0_j + t * phi1(-t L) b1_j ,     phi1(z) = sum_k z^k / (k+1)!
```

The checker accepts `max|y - y_ref| <= atol + rtol * max|y_ref|` (mixed
absolute/relative infinity norm over the whole vector, not per component).

## Algorithm

`L` is symmetric and positive semidefinite, so `-t L` is symmetric negative
semidefinite and both `exp(-tL)` and `phi1(-tL)` are real symmetric matrix
functions of `L`.  The implementation is an **adaptive symmetric Krylov
(Lanczos) method**.

1. **Projection.** A Lanczos run on `A = -L` starting from the right-hand side
   `b` produces an orthonormal basis `V` and the symmetric tridiagonal
   projection `T = V^T A V`.  Full reorthogonalisation is used: the residual
   of every step is projected against the whole basis, with a second pass
   whenever the projection removes a large part of the residual (DGKS
   criterion).  This keeps `||V^T V - I||` at machine precision, which is
   essential because the projected matrix function is evaluated spectrally.
   The projection is done in cache-sized chunks: the arithmetic is identical to
   one BLAS-2 call, but the second pass over each chunk is served from cache,
   which roughly halves the memory traffic of the orthogonalisation.

2. **Spectral evaluation of the projected function.** The tridiagonal matrix is
   diagonalised with `scipy.linalg.eigh_tridiagonal` (O(m^2), no dense
   eigenvector matrix for the full space) and the two matrix functions are
   evaluated at the Ritz values:
   * `exp(-tL)`     -> `f(theta) = exp(t*theta)` (`theta <= 0`, no overflow);
   * `t phi1(-tL)` -> `f(theta) = t * phi1(t*theta)` with `phi1(z) =
     expm1(z)/z` for `z <= -1/2` and the factorial series `sum_k z^k/(k+1)!`
     near zero.  Both branches are free of cancellation.

   Because the Lanczos start vector is `b`, the projected right-hand side is
   `V^T b = ||b|| e_1`, so only the first row of the eigenvector matrix is
   needed: `f(T) u = Q (f(theta) * (||b|| Q[0,:]))`.

3. **Adaptive Krylov degree.** The degree is not fixed a priori.  The Krylov
   run is extended in geometric checkpoints (factor 1.5) and the approximation
   at the current degree is compared with the previous one; the run stops when
   the two agree to within 5 % of the tolerance `atol + rtol * S`, where `S` is
   the magnitude of the computed vector, shared between the two terms of a
   request so that both are measured against the same scale.  A vanishing
   current scale is explicitly *not* accepted as agreement: for large `t` the
   intermediate approximations underflow to zero until the Krylov space has
   resolved the slow modes, and that degenerate agreement carries no
   information.

4. **Diagonal operators.** If `L` is diagonal the action is purely local and
   is evaluated element-wise in O(n) — no Krylov process at all.

5. **Time budget.** The session wall-clock budget (15 s for `one_shot`,
   150 s for `reuse`) is measured from the moment the header arrives.  The
   remaining time is shared between the outstanding requests and between the
   two terms of a request, and the Lanczos loop stops at its deadline, so the
   process always answers and exits cleanly.

## Files

* `solve.py` — the whole solver and the wire protocol (self-contained).
* `run.sh` — entry point; forces single-threaded BLAS so that the CPU budget
  equals the wall-clock budget, then executes `solve.py`.

No build step is required (pure Python + NumPy/SciPy from the frozen image);
there are no downloads, no generated code and no JIT compilation at run time.
`run.sh` uses only relative paths and can be executed from any working
directory.

## Dependencies

Python 3.12, NumPy 2.5, SciPy 1.18 (all pre-installed in the frozen image).
Only `numpy`, `scipy.linalg` and `scipy.sparse` are used.

## Public checks

```sh
python tools/run_local.py --manifest samples/manifest.json \
       --command "./run.sh" --repeats 3 --output dev_report.json
```

All six public sample cases (five synthetic structure families plus the
analytic two-point control) pass with `status = ok`; the control case is
checked at the tightest accuracy level `p10`.

Further checks performed during development:

* the byte protocol (READY line, header, CSR arrays, request/response pairs,
  exit code 0) was exercised through the official local runner;
* accuracy was verified against an independent dense reference
  (`numpy.linalg.eigh` of the same matrix, with `exp` and `phi1` evaluated
  spectrally) on synthetic 2-D diffusion operators with `t*||L||_inf` from
  1e2 to 1e5: the observed errors were 1e-7 to 1e-3 of the allowed tolerance;
* `t = 0`, singular `L`, zero right-hand sides, diagonal `L` and the `n = 1`
  corner case are handled explicitly;
* timing was measured for synthetic 2-D diffusion operators up to
  `n = 1.2e5` (`nnz ~ 6e5`, `t*||L||_inf = 1e5`) in both `one_shot` and
  `reuse` mode; the Krylov degree actually used is reported on stderr for
  every solve.
