# Exponential propagation solver (FAMEXP2)

Computes `y = exp(-tL) b0 + t*phi1(-tL) b1` for a sparse symmetric positive
semidefinite `L`, one or sixteen requests per session, over the binary
stdin/stdout protocol of `API.md`.

## Files

- `run.sh` – entry point: runs `solver.py` with `/usr/local/bin/python3` (the
  interpreter that has NumPy/SciPy; `/usr/bin/python3` in this image does not),
  otherwise the first interpreter on PATH that imports NumPy and SciPy;
  single-threaded BLAS.
- `solver.py` – protocol I/O, method selection, Lanczos driver, stopping logic.
- `kernels.c` – native kernels loaded with `ctypes`: CSR Lanczos steps (single
  vector, and two vectors fused into one matrix pass), Chebyshev evaluation of
  `f(T)e1` for a tridiagonal `T`, and a one-pass matrix statistics routine
  (infinity norm, Gershgorin lower bound, ordering locality, diagonal test).
- `build.sh` – compiles `kernels.c` into `build/libk_base.so` (generic x86-64)
  and `build/libk_avx2.so` (AVX2/FMA). Both use `-ffp-contract=off` so the fused
  and single-vector kernels produce bitwise identical Lanczos vectors (required
  by the basis-regeneration path). Prebuilt copies are included; at start-up
  the solver loads the AVX2 build if `/proc/cpuinfo` reports avx2+fma, else
  the generic one. If neither loads it compiles `kernels.c` into /tmp before
  printing READY (input independent); if that also fails the same algorithm
  runs with NumPy/SciPy only.

## Dependencies

Python 3, NumPy, SciPy (`eigh_tridiagonal`, `ive`, `dct`,
`reverse_cuthill_mckee`), gcc for the native kernels. No network access or
extra packages.

## Algorithm

1. Exact special cases: `t = 0` (y = b0), zero matrix (y = b0 + t b1),
   diagonal matrix (elementwise formulas), `n <= 200` (dense eigendecomposition).
   With the Gershgorin lower bound `lam_low <= lambda_min`, the `b0` part is
   dropped when `exp(-t*lam_low)*||b0|| <= 0.05*atol`, and the `b1` part when
   `t*||b1|| <= 0.05*atol` (both are rigorous norm bounds).
2. Otherwise each nonzero right-hand side gets its own symmetric Lanczos
   recurrence (`b0` with `exp(-t x)`, `b1` with `t phi1(-t x) = (1-exp(-t x))/x`)
   with the basis kept in memory. When both are nonzero the two recurrences
   advance in lockstep in a fused kernel that streams the matrix once and
   gathers from an interleaved copy of the two current vectors.
3. If the matrix ordering is scattered (mean |i-j|/n > 0.02) and the expected
   amount of work is large, the matrix is reordered once per session with
   reverse Cuthill–McKee (vectors are permuted in and out) to improve the
   locality of the gathers.
4. Checkpoints evaluate `c = ||b|| f(T_k) e1`. Intermediate checkpoints use a
   Chebyshev series on `T_k` (interval from bisection on `T_k`, O(k·degree));
   the coefficients used to form the solution always come from the
   tridiagonal eigendecomposition (`eigh_tridiagonal`), which is also used for
   checks once the estimate approaches the Chebyshev rounding floor.
5. Stopping. The change between consecutive checkpoints is bounded by
   `||V dc||_inf <= ||dc||_1` (unit-norm basis vectors). A part stops when this
   is below a quarter of `atol + rtol*M`, `M` a lower estimate of `max|y|`;
   when only an optimistic test passes, the exact infinity norms of the change
   and of `y` are computed with one GEMV and tested against the same
   threshold. Safeguards against premature acceptance:
   - for `exp`, the Krylov space must have reached the active part of the
     spectrum (`t * min Ritz value <= 60`), otherwise iterates that are both
     ~0 could look converged (e.g. isolated vertices or slow low modes);
   - before the regular schedule, two consecutive checks must pass.
   The schedule and an iteration cap come from the a-priori Chebyshev degree
   for `exp(-x)` on `[0, t||L||inf]` (the cap uses the absolute tolerance, at
   which Lanczos is guaranteed to have converged).
6. Memory: stored bases are capped at 2.4 GB in total (buffers are pooled across requests); rows beyond the cap
   live in a ring buffer and are regenerated deterministically when the
   solution is assembled.
7. I/O: input is read with `readinto` straight into NumPy buffers, pipe
   buffers are enlarged with `F_SETPIPE_SZ`, outputs are written raw, and the
   process leaves with `os._exit(0)` right after the last result. Any
   exception in the main path falls back to SciPy `expm_multiply` on the
   augmented matrix.

## Public checks

`python tools/run_local.py --manifest dev/manifest.json --command ./run.sh`
and the same for `dev_full/manifest.json`: all 56 dev and 40 dev_full cases
pass, worst error 0.053 of the tolerance. Additional synthetic stress sets
(dense-eigendecomposition references; disconnected graphs with isolated
vertices, heavy-tailed hubs, potentials, signed incidence Laplacians, paths,
2x2 log-spectrum blocks, a 1e6-stiff edge, Erdős–Rényi; all RHS categories
including zero and near-zero vectors; t·||L|| from 1e-2 to 1e5) pass with worst
error 0.07 of the tolerance, also with reordering forced on, with the basis
cap forced down to 8 rows (regeneration path), and with the NumPy-only
fallback. A synthetic maximum-size case (random geometric graph, n = 120000,
nnz = 979908, t·||L||inf = 1e5, p10, both right-hand sides nonzero) runs in
3.4 s / 2.43 GB as one_shot and 42 s / 2.44 GB as a 16-request reuse session
(limits 60 s / 240 s / 4 GiB), with error 0.014 of the tolerance against a
1000x tighter solve.
