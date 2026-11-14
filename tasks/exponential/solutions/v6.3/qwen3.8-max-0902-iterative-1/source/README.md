# FAMEXP2 — sparse exponential propagation

Computes, for a real symmetric positive semidefinite sparse operator `L` given in
CSR, a time `t >= 0` and one or sixteen right-hand side pairs `(b0, b1)`,

```
y = exp(-t L) b0 + t*phi1(-t L) b1 ,      phi1(z) = (e^z - 1)/z ,  phi1(0) = 1
```

i.e. the state at `s = t` of `u' = -L u + b1`, `u(0) = b0`.  Every component of `y`
is returned; `L` may be singular, `t` may be zero, `b0`/`b1` may be zero.

## Files

| file | role |
|---|---|
| `run.sh` | entry point executed by the evaluator |
| `solve.py` | protocol driver, spectral interval, coefficient generation, dispatch |
| `fastcore.c` | fused hot loops (norm bound, λmax probe, Chebyshev recursion) |
| `build.sh` | offline build of `libfastcore.so` (plain `gcc -O3`, no dependencies beyond libm) |
| `libfastcore.so` | prebuilt copy of the kernel as a fallback if the build step is skipped |

Dependencies: Python 3.12 and NumPy, plus a C compiler for `build.sh` (plain
`gcc -O3 -mtune=native`, libm only — no ISA-specific flags, so the object stays
portable across x86-64 cores).  `run.sh` re-tries the compile only if the shared
object is missing, which happens before `READY` and is therefore not charged to
the session clock.  If no compiler exists at all, `solve.py` falls back to a pure
NumPy/SciPy evaluation of the same algorithm.  SciPy is *not* imported on the
fast path: importing it costs ~3 ms of process teardown at exit, which is inside
the measured session, so the DCT-I is done with `numpy.fft.rfft` on the even
extension instead.

## Algorithm

Everything reduces to two scalar functions of the eigenvalues of `L`,

```
f1(λ) = exp(-t λ)          f2(λ) = t*phi1(-t λ) = (1 - exp(-t λ))/λ ,  f2(0) = t
```

and `y = f1(L) b0 + f2(L) b1`.  Because `L` is symmetric with a spectrum inside a
known real interval, both are evaluated by a single **Chebyshev expansion on
`[λlo, λhi]`**, using the three-term recursion

```
T_0 = I ,  T_1 = x̂ ,  T_{k+1} = 2 x̂ T_k - T_{k-1} ,   x̂ = (L - c I)/ρ
c = (λhi+λlo)/2 ,  ρ = (λhi-λlo)/2
y = Σ_k ( a1_k T_k b0 + a2_k T_k b1 )
```

Both right hand sides are advanced in the same pass over the operator, so one
matrix traversal serves the two expansions.

Why not Krylov: for the stiff end of the input range (`t·||L||∞` up to 1e5) the
required polynomial degree is `~sqrt(t·λhi·ln(1/rtol))`, i.e. 1e3–2e3 matvecs.
Lanczos needs a comparable dimension there, and its orthogonalisation costs
`O(n·D²)` memory traffic — an order of magnitude more than the matvecs it saves
at `n = 1e5`.  Chebyshev needs no orthogonalisation, no factorisation, has no
breakdown, and its truncation error is bounded a priori by the coefficient tail.
For the whole documented input range it is the cheapest robust option.

### Spectral interval

* `λhi` starts from the rigorous Gershgorin bound `||L||∞ = max_i Σ_j |L_ij|`
  (one streaming pass over `data`, no index access).
* When the predicted degree is large (`M >= 150`), 24–32 Lanczos steps produce a
  tighter estimate and `λhi = 1.05 · λmax_est` is used instead; this removes the
  factor `sqrt(||L||∞/λmax)` (up to 1.4×) that Gershgorin costs on graph
  operators (measured 1.13–1.99× on the local families).  `λlo = -1e-8 λhi` absorbs rounding in a nominally singular `L` at no cost in degree.
* The estimate is never trusted blindly: the kernel monitors `max|T_k(x̂)b|`,
  which stays `<= ||b||_2` while the spectrum is inside the interval and grows
  exponentially as soon as it is not.  On blow-up (or any non-finite output) the
  driver recomputes with the certified `||L||∞` interval, and a wider interval
  plus the NumPy path as a last resort.

### Degree selection

Coefficients come from Chebyshev interpolation at `M+1` points (DCT-I), `M`
being a generous upper bound
`M ≈ 1.12 sqrt(t(λhi-λlo)(ln(1/rtol) + ½ln n + 6)) + 32`.
`tail1[k] = Σ_{j>k}|a1_j|` and `tail2[k]` are precomputed once per session.  The
kernel stops at the first `k` where the truncation error is provably below
`safety·(atol + rtol·||y_k||∞)`, `safety = 0.35`, using whichever of two bounds
is satisfied first:

* rigorous: `tail1[k]·||b0||_2 + tail2[k]·||b1||_2` (since `||T_j(x̂)||_2 <= 1`);
* sharp: `(tail1[k]·m0 + tail2[k]·m1)·6` with `m0, m1` the largest
  `||T_j(x̂)b||_∞` observed so far — the 2-norm bound is pessimistic by
  `~sqrt(n)` for delocalised vectors.

Low-`η` sessions therefore stop after 2–10 matvecs instead of always paying for
the worst case, and `atol` alone keeps the criterion meaningful when `||y||` is
tiny.

### Special cases

`t = 0` → `y = b0`; zero operator (`nnz = 0` or `||L||∞ = 0`) → `y = b0 + t b1`;
`b0 = 0` or `b1 = 0` → the kernel specialises to a single column, which halves
the gather traffic; both zero → `y = 0`.

## Interface and resource notes

* `READY\n` is emitted after input-independent initialisation only: imports plus
  oversized pre-faulted buffers (input, RHS, output, workspace).  Page faults for
  ~33 MB would otherwise land inside the measured session.
* The wire bytes are read with `os.readv` straight into those buffers;
  `F_SETPIPE_SZ` is raised to 1 MiB to cut transfer round trips.  `indptr`,
  `indices` and `data` are used as zero-copy views in the exact wire layout
  (`int64` / `int32` / `float64`), so the operator is never copied or converted.
* Output is written from a memoryview and flushed per request; the process ends
  with `os._exit(0)` right after the last response.
* Rows are processed in blocks of 2048 inside the kernel, which keeps the
  rotating Chebyshev vectors cache resident on banded and clustered operators
  (≈10 % on the local network/mesh-with-potential cases, neutral elsewhere).
* Single threaded throughout (one logical CPU); peak RSS ≈ 130 MB at `n = 1e5`,
  4 GiB limit untouched.  Session state reused across the 16 requests of a
  `reuse` session: interval, coefficients and tails are computed once.

## Local checks performed

* `tools/run_local.py --manifest samples/manifest.json` — all 6 public cases
  (5 synthetic families + the analytic control) pass with 3 repeats.
* Edge cases built locally and checked against dense `eigh`: zero operator,
  `t = 0`, `n = 1`, zero/one-sided right-hand sides, singular path Laplacian,
  positive off-diagonal entries, diagonal potential with a spectral gap,
  `||L||∞ = 5e7` with `t = 1e-10`, log-spaced spectrum with `t·||L||∞ = 1e5`,
  and a 16-request lifecycle mixing zero and non-zero RHS.
* Scale checks at `n ≈ 1e5`, `nnz ≈ 2.5e5`–`1e6`, `t·||L||∞ ∈ {1e-2 … 1e5}`,
  `p4`/`p7`/`p10`, one_shot and reuse, on 2-D diffusion meshes, kNN geometric
  graphs, preferential-attachment networks, a log-spaced spectrum and a mesh with
  a diagonal potential: worst observed error `5.2e-2` of the acceptance
  threshold over 26 local cases, all sessions inside the wall and CPU budgets
  (worst reuse session 22 s of the 150 s cap).
* Interval-tightening failure path exercised by forcing the margin below 1
  (`FAMEXP_MARGIN=0.90/0.98`): the blow-up monitor fires, the driver recomputes
  with the certified Gershgorin interval, and the answers stay correct.
