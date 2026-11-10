# Exponential propagation solver (FAMEXP2)

Computes `y = exp(-tL) b0 + t*phi1(-tL) b1` for a sparse symmetric PSD `L`
over the FAMEXP2 stdin/stdout protocol (one-shot and 16-request reuse sessions).

## Algorithm
* **Augmented Chebyshev series.** `y` is the first block of `exp(-tB)[b0; 1]`
  with `B = [[L, -b1], [0, 0]]`, whose spectrum is spec(L) plus 0, i.e. inside
  `[0, lam]`.  `exp(-t x)` on `[0, lam]` is expanded in Chebyshev polynomials
  (coefficients `exp(-rho) I_k(rho)`, `rho = t*lam/2`, from Miller's backward
  recurrence).  The scalar block of the augmented iterate is `(-1)^k`, so every
  term costs one fused SpMV + vector update (`v_{k+1} = 2(I - aL)v_k - v_{k-1}
  + 2a b1`, `y += 2 e_{k+1} v_{k+1}`), with no Krylov basis to store.
* **Spectral bound `lam`.** Rigorous Gershgorin and Collatz-Wielandt
  (`x = sqrt|diag|`) bounds; for long series a short Lanczos run (35-60 steps,
  1% + 2*residual margin), capped by the rigorous bound.  If an iterate ever
  exceeds 1e4 times its theoretical size (lam too small), the request is
  restarted with the rigorous bound.
* **Adaptive truncation.** Stop when the coefficient tail times the observed
  iterate size, plus the k^2-growth bound of the phi1 part, is below 2% of
  `atol + rtol*max|y|` (in practice errors are <= 4% of the tolerance).
* **Small components.** If every connected component has <= 24 nodes (e.g.
  block-diagonal operators) each block is diagonalised once (Jacobi) and every
  request is evaluated exactly in O(n).
* **Trivial cases.** `t = 0` or `L = 0`: `y = b0 + t*b1`.
* **Performance.**  C, single thread.  Long runs use a SELL-4 copy of `L`
  (4 rows per slice, length-sorted in windows of 128 rows, column-major), which
  removes per-row branch mispredictions and gives 4 independent FMA chains;
  poorly ordered matrices (e.g. random geometric graphs) are first reordered by
  reverse Cuthill-McKee.  Input is read straight into a 2 MB-page arena that is
  pre-faulted before `READY`; pipes are enlarged to 1 MB; the Gershgorin scan
  runs on rows as their values arrive and the Collatz-Wielandt pass runs while
  the first right-hand side streams in.  Reuse sessions keep `lam`,
  coefficients, orderings and buffers across requests.

## Files / dependencies
* `expo.c` - complete solver (C99, libm only).
* `expo` - prebuilt binary: `gcc -O3 -march=x86-64-v3 -o expo expo.c -lm`.
* `run.sh` - starts the prebuilt binary; if it is missing or the CPU lacks
  AVX2/FMA it recompiles into `/tmp` with the system gcc before `READY`.
No other dependencies; no build step is required.

## Public checks
* `tools/run_local.py` on `dev/` (56 cases incl. reuse and controls) and
  `dev_full/` (40 full-size cases): all pass, worst error 0.034 of the
  tolerance.
* Extra synthetic checks against dense eigendecompositions: random geometric
  graph at t*||L|| = 1e5 (RCM + SELL + Lanczos path, one-shot and reuse),
  mixed giant/small components with a potential, power-law hub graph, 3x3
  log-spectrum blocks, 1e9-scaled Laplacian, explicit zero matrix, n = 1, and
  a forced lam underestimate (restart path).
* Public evaluator (`submit --test`): all 72 cases valid, public-set score
  62.8-63.7 across runs (baseline 50); small cases are quantised by the
  harness to ~10 ms steps.
* Stress: n = 120000, nnz ~ 1e6, t*||L|| = 1e5, p10, 16 requests: 18 s wall,
  ~200 MB peak memory.
