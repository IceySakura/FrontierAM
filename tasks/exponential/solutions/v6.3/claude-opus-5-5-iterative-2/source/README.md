# Exponential propagation solver (FAMEXP2)

Computes `y = exp(-tL) b0 + t*phi1(-tL) b1` for a sparse symmetric PSD matrix `L`
(one or 16 sequential right-hand-side pairs per session).

## Files
- `run.sh` – entry point.  Runs the native solver `expsolve_v3` (x86-64-v3: AVX2/FMA) when the
  CPU supports it (CPUID check via `expsolve_gen --has-v3`, or `/proc/cpuinfo`), else
  `expsolve_gen` (generic x86-64), else the Python fallback `solver.py`.
- `expsolve.c` – the native session driver: raw `read`/`write` protocol I/O, JSON header
  parsing, strategy selection, Chebyshev coefficients, spectral bounds, reordering.
- `expkern.c` – numerical kernels (also built as a ctypes library for `solver.py`).
- `chebtmpl.h`, `selltmpl.h`, `sell2i.h` – kernel templates (CSR / pattern / 16-bit-offset
  variants, SELL-4 AVX2 variants; one or two right-hand sides, the latter also with
  interleaved vector storage).
- `solver.py` – Python/NumPy/SciPy fallback implementing the same basic algorithm
  (Chebyshev + Lanczos bound + block path) through `libexpkern_*.so`.
- `build.sh` – rebuilds all binaries from source with gcc (the prebuilt binaries shipped in
  this directory are replaced only when compilation succeeds).

## Dependencies
gcc and libm for building (prebuilt binaries included).  The native solver needs only libc/libm.
The fallback needs Python 3 with NumPy and SciPy.  No network, no other packages.

## Algorithm (native solver)
1. `t == 0` -> `y = b0`;  `L == 0` -> `y = b0 + t b1`.
2. Small independent blocks.  If every row has at most one nonzero off-diagonal partner (blocks
   of size <= 2, e.g. diagonal or 2x2-rotated operators), one sequential pass verifies it and each
   pair gets closed-form 2x2 matrices `V diag(exp(-t lam)) V^T` and `V diag(g(lam)) V^T`; each request
   is then a 4-term stencil.  More generally, if every connected component of the off-diagonal
   pattern has at most 24 nodes (or `n <= 64`) and a cost model prefers it, each block is diagonalised exactly
   (closed form for 2x2, cyclic Jacobi otherwise) and applied with `exp(-t lam)` and
   `-expm1(-t lam)/lam` (= `t` at `lam = 0`).  The component search is skipped when the
   polynomial work is too small for this path to pay off.
3. Otherwise a Chebyshev expansion on `[lo, hi]`, `lo = 0` (PSD; tiny negative margin for
   round-off), `hi` = the best available upper bound on `lambda_max`:
   - Gershgorin (`||L||_inf`);
   - rigorous Collatz-Wielandt bound `lambda_max(L) <= rho(|L|) <= max_i (|L|x)_i / x_i` after a few
     power steps on `|L|`;
   - an 8-step Lanczos run gives `theta_1 <= lambda_max`; if the CW bound may be loose, a longer
     Lanczos run gives the Kuczynski-Wozniakowski probabilistic bound `theta_1/(1-eps)`
     (failure probability 1e-2), with the number of steps chosen to balance its cost.
   Coefficients: `exp(-tx)` via scaled Bessel functions (Miller's backward recurrence,
   normalised by `I_0 + 2 sum I_k = e^beta`); `(1-exp(-tx))/x` via a DCT (radix-2 FFT) of samples
   at Chebyshev points, truncated at its round-off floor.
4. Matrix preparation (when enough iterations are expected): Cuthill-McKee-style BFS ordering
   followed by sorting rows by length inside windows of 1024 rows (keeps locality and makes row
   lengths predictable); unweighted Laplacians (all off-diagonals equal) are stored as
   pattern + diagonal; banded results use an AVX2 SELL-4 layout, others CSR (16-bit column
   offsets when the bandwidth allows).
5. Fused kernel: each sweep over the matrix advances two Chebyshev steps (the second lags by
   the bandwidth so its rows are still in cache), updates the recurrence and the accumulated
   result, and tracks norms.  `b0` and `b1` share one pass when both are nonzero (with
   interleaved storage in the SELL-4 kernel when rows have at least 5 entries on average).
6. Adaptive degree: stop once `10 * tail(|coef|) * max_j ||T_j(S) b||_inf <= atol + rtol ||y||_inf`.
7. Safety net: if `||T_j(S) b||_2 > 1.25 ||b||_2` (impossible when the spectrum lies in the
   interval) the request is recomputed with a wider interval (x1.1, then Gershgorin, then
   Gershgorin with growing round-off margins below 0).
All allocations above 2 MiB use transparent huge pages; denormals are flushed to zero; the
process exits with `_exit(0)` right after the last result is written.

## Public checks
- `tools/run_local.py` on `samples/` (all pass).
- Locally generated cases (n = 50k-100k, nnz up to 1M): diffusion (9-point, high-contrast
  coefficients), weighted/unweighted geometric graphs, heavy-tailed and hub-dominated networks,
  signed Laplacians with potentials, 2x2-rotated log-spectrum operators; `t*||L||_inf` from 1e-2
  to 1e5; p4/p7/p10; one-shot and reuse; references from a long-double Chebyshev evaluation.
  All pass with errors <= 0.1 of the tolerance.
- An adversarial suite against dense eigendecompositions (isolated vertices, empty rows,
  explicit zeros, mixed block sizes, star graph, path graph, tiny t, huge norms, zero vectors,
  n = 1 and 3, control-style cases: zero matrix, t = 0, diagonal, positive off-diagonals, singular
  2x2 pairs, with one-shot and 16-request sessions).  All pass, also with the generic build and
  with the spectral bound deliberately underestimated (exercising the divergence fallback).
- Worst-case timing: n = 120000, nnz = 1e6 expander, t*||L|| = 1e5, p10, both vectors nonzero,
  16 requests: about 37 s (limit 150 s), 113 MB.
