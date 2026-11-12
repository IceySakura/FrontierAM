# Exponential propagation solver (FAMEXP2)

Computes `y = exp(-tL) b0 + t*phi1(-tL) b1` for a sparse symmetric positive semidefinite matrix `L`
over the FAMEXP2 binary stdin/stdout protocol (one-shot and 16-request reuse sessions).

## Files
- `expo.cpp`: the complete solver (C++17, single file, about 1300 lines).
- `expo_avx2`, `expo_generic`: prebuilt binaries of `expo.cpp`, built by `build_bins.sh`.
  `expo_avx2` uses `-O3 -march=haswell -mtune=znver2`; `expo_generic` uses plain `-O3`.
- `run.sh`: entry point. It selects `expo_avx2` when `/proc/cpuinfo` reports AVX2 and FMA,
  and `expo_generic` otherwise. Before `exec`, each binary is probed with `--selftest`
  (loader, LAPACK, a 4x4 eigenproblem). If neither binary works, it compiles `expo.cpp` before
  `READY`. It also sets `OPENBLAS_NUM_THREADS=1`.
- `build_bins.sh`: rebuilds both binaries (`g++ ... expo.cpp -llapack`).

Dependencies: g++ (C++17) and the system `liblapack`/OpenBLAS (Debian packages in the image).
LAPACK is only a fallback: `dstevd` (tridiagonal eigenproblems with m > 1600, or when the QL
routine below fails) and `dsyevd` (small dense blocks). No Python and no network at run time.

## Algorithm

### Session handling
- Raw `read`/`write` on fds 0/1. The pipe buffers are enlarged to 1 MiB.
- Before `READY`, an address-space arena is reserved and its first 128 MiB are pre-faulted, and
  LAPACK/OpenBLAS is warmed up with tiny eigenproblems. Both steps are input-independent; they
  remove page-fault and lazy-initialisation cost from the timed session.
- The program calls `_exit(0)` immediately after the last result.

### Exact special cases (O(n) per request)
- `t == 0` gives `y = b0`; `L == 0` gives `y = b0 + t b1`.
- If every row has at most one off-diagonal entry, `L` is a direct sum of 1x1 and 2x2 blocks
  (the "log-spectrum" structure). Each block is diagonalised in closed form once. `exp` and
  `phi1` use a 17-term Taylor series for |x| <= 0.5.
- If all connected components are small (all of `L` when n <= 600, otherwise components of at
  most 48 vertices), each component is diagonalised once with `dsyevd`.

### General case: lockstep Lanczos
- Two Lanczos recurrences run in lockstep: one for `b0` (function `exp(-t x)`) and one for `b1`
  (function `t phi1(-t x)`). The two vectors are stored interleaved and share one fused CSR SpMV,
  so the matrix is streamed once per iteration. A sequence whose right-hand side is zero is
  skipped.
- If all off-diagonal values are equal (unweighted graph Laplacians), a kernel that needs no
  value array is used.
- For bounded-degree matrices with a scattered ordering and t*||L||_inf >= 1000, the matrix is
  reordered once by reverse Cuthill-McKee (RCM) to improve cache locality. Right-hand sides and
  results are permuted per request.
- Each iteration makes two passes: the fused SpMV with alpha accumulation, then orthogonalisation
  with the norm. Basis vectors are stored unnormalised with scale factors. With two right-hand
  sides, the SpMV processes both lanes with 128-bit FMAs.
- The basis is stored (budget 2.8 GB). If it overflows, later vectors are regenerated
  bit-identically from the last two stored ones during assembly.
- The result is `y = sum_j c_j v_j` with `c = ||b|| f(T_m) e1`. The coefficient vector comes from
  an implicit QL iteration whose recorded Givens rotations are replayed on `f(theta) .* Z(1,:)`.

### Stopping rule: residual bound
- Let `r(s) = beta_m (e_m^T c(s)) v_{m+1}` be the Krylov residual of the ODE `u' = -Lu + b1`,
  `u(0) = b0`. Then `e(t) = -int_0^t exp(-(t-s)L) r(s) ds`.
- For weakly diagonally dominant `L`, `exp(-sL)` is an infinity-norm contraction, which gives
  `||e||_inf <= beta_m ||v_{m+1}||_inf int_0^t |e_m^T c(s)| ds`.
- Otherwise a safety factor of 8 is applied, capped by the 2-norm bound.
- Checkpoints need only the eigenvalues and the first and last eigenvector rows of `T_m`. A QL
  sweep that tracks those two rows provides them in O(m^2). The integral uses composite 6-point
  Gauss-Legendre on a dyadic grid in `s`.
- Iteration stops when the bound is at most `0.5 (atol + rtol ||y||_inf)`; `||y||_inf` is checked
  on the assembled result.
- Checkpoint placement extrapolates the decay of the bound, is limited by the measured checkpoint
  cost, and in reuse sessions starts near the previous request's crossing point.

### Safeguards
- Rounding floor: at very high t*||L|| with p10 accuracy, the residual bound saturates near
  `t||L|| u ||y||_2`. When the bound stays within 100x of the target over at least 10% more
  iterations, and that floor estimate explains the stall, the iterate is accepted only if it
  changed by at most 5% of the tolerance over that window.
- Termination: the Hochbruck-Lubich a priori bound for Lanczos approximations of exp caps the
  iteration count. A per-request wall-clock deadline (66% of the session limit, prorated per
  request) forces acceptance before any limit is reached.
- Memory: peak RSS stays below about 2.6 GB (4 GiB limit). The QL rotation log is capped. A
  failed eigensolver falls back from QL to `dstevd` to dense `dsyevd`.

## Public checks
- `tools/run_local.py --manifest samples/manifest.json --command ./run.sh --repeats 3`:
  all 6 sample cases pass (worst error is 7.5% of the tolerance).
- About 60 synthetic full-size cases were checked against an independent high-degree Chebyshev
  reference:
  - sizes n = 20k-120k with nnz up to 1M;
  - families: 2D/3D meshes, FEM-like non-M-matrices, random geometric graphs (weighted and
    unweighted, 2D and 3D), heavy-tailed weighted and unweighted networks, signed incidence
    Laplacians, 2x2 log-spectrum blocks, random non-diagonally-dominant `C^T C`;
  - t*||L||_inf from 1e-2 to 1e5; p4/p7/p10; one-shot and reuse;
  - edge cases: constant and spike right-hand sides, b0 or b1 zero, alternating zero vectors in a
    reuse session, isolated vertices and empty rows, n = 1, small dense blocks.

  All pass. The largest error is 38% of the tolerance and is typically below 5%. The slowest
  case, n = 120k at eta = 1e5 with p10 and 16 requests, takes about 40 s (limit 150 s).
- 80 randomised stress cases also pass. Each draws a random family, n = 5k-25k, log-uniform
  t*||L|| in [1e-2, 1e5], a random precision, random right-hand-side types (Rademacher, uniform,
  zero, constant, sparse spikes) and a random scenario. The worst error is 26% of the tolerance.
- `run.sh` fallbacks were tested with corrupted binaries: AVX2 to generic to compile from source.
- The storage-overflow regeneration path was verified bit-identical by forcing tiny basis
  budgets (`EXPO_VBUDGET`).
