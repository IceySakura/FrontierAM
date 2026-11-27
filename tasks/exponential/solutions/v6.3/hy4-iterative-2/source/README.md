# Exponential propagation (FAMEXP2)

## What the program computes

For a real symmetric positive semidefinite sparse matrix `L`, a time `t` and a
pair of right-hand sides `(b0, b1)`, each request is answered with

```
y = exp(-t L) b0 + t phi1(-t L) b1 ,   phi1(z) = (exp(z)-1)/z .
```

The implementation is a single C++ program (`solver.cpp`), compiled by
`build.sh` (and, as a fallback, by `run.sh` if the binary is absent).  It uses
only the C/C++ standard library and libm; no Python packages, BLAS/LAPACK or
network access are needed at run time.

## Algorithm

Three strategies are chosen from the input matrix and the scaled time
`eta0 = t*||L||_inf`:

1. **Exact small blocks.** When `eta0 >= 20` and the graph of `L` splits into
   connected components of size at most 96 with a bounded total cubic cost
   (`sum k^3 <= 5e6`), each block is diagonalised with a cyclic Jacobi
   eigensolver in 80-bit extended precision.  The two scalar functions are
   evaluated at the eigenvalues and applied to both right-hand sides in double
   precision.  This is O(nnz + n k) per request and is accurate to working
   precision even for eigenvalues many orders below the spectral radius; it
   covers the diagonal / log-spectrum family.  Component detection aborts as
   soon as one component exceeds the cap, so connected operators pay almost
   nothing for this check.  For small `eta0` a few cheap polynomial steps are
   faster than any factorisation, so the check is skipped.

2. **Chebyshev expansion on `[0, B]`.** `B >= lambda_max(L)` is a rigorous,
   tight upper bound from a scaled Gershgorin (Collatz-Wielandt) bound on the
   entrywise absolute matrix `|L|`: a short power iteration on `|L|` supplies a
   positive vector `s`, and `B = max_i (|L| s)_i / s_i`, capped by the ordinary
   row-sum bound.  With `S = 2 L / B - I` (spectrum in `[-1,1]`) and `z = t B/2`,
   the coefficients of `exp(-t lambda)` are
   `a_0 = s_0`, `a_k = 2 (-1)^k s_k`, where `s_k = exp(-z) I_k(z)`, obtained in
   O(K) by Miller's stable backward recurrence normalised with
   `I_0(z) + 2 sum_{k>=1} I_k(z) = exp(z)`.

   For `t phi1(-t lambda) = integral_0^t exp(-tau lambda) d tau`, the exact
   tail-sum identity
   `M_k = sum_{j>k} (j-k) s_j`,
   `c_0 = (4/B) M_0`, `c_k = (8/B) (-1)^k M_k`
   gives the coefficients.  Both `M_k` and the coefficient tails
   (`sum_{k>m}|a_k|`, `sum_{k>m}|c_k|`) come from one stable O(K) downward
   sweep; no quadrature, special functions or dense transforms are needed.

3. **Adaptive evaluation.** A forward Chebyshev three-term recurrence
   accumulates both series simultaneously (one sparse traversal per degree,
   interleaved columns).  After each degree it compares the analytic
   coefficient tail with the acceptance
   threshold `atol + rtol*max|y|` formed from the current partial sum, and
   stops once the tail is below 2% of that threshold for two consecutive
   degrees.  This adapts the degree to the actual output scale and precision
   level.  A column whose RHS is exactly zero is omitted, so the common
   `b0 = 0` case costs one column.  For large matrices (`n >= 10000` and
   `nnz >= 150000`) a symmetry-exploiting upper-triangle matvec (one pass over
   half the entries, averaged over both stored triangles) replaces the full
   CSR traversal for one-column cases.  For `t*||L|| <= 1e-6` a short
   Taylor/Horner expansion of the ODE state is used; `t = 0`, `L = 0` and zero
   right-hand sides have direct paths.

All session-specific work (matrix conversion, component detection or spectral
bound, coefficients, requests) is inside the measured interval; the process
prints `READY\n` before reading any input and writes exactly `8*n` bytes per
request.  No output other than the READY line and the result vectors is
produced.

## Files

- `solver.cpp` — complete source of the streaming solver.
- `build.sh` — offline build (`g++ -O3 -std=c++17 -funroll-loops`).
- `run.sh` — executable entry point; execs the prebuilt `solver`, rebuilding
  it only if missing.
- `solver` — prebuilt amd64 binary produced by `build.sh`.

## Dependencies

Linux amd64, a C++17 compiler (g++ 12 in the frozen image), libc/libm and
libstdc++.  No BLAS, LAPACK, Python, SciPy or network access at run time.  The
program is single-threaded and respects inherited CPU affinity.

## Public checks

The six reduced-size sample cases were run with
`python tools/run_local.py --manifest samples/manifest.json --command "./run.sh" --repeats 3`:
all pass (including the analytic control) on three repeats each, with medians
between 0.4 ms and 10 ms on the development core.  Additional development
checks used dense symmetric eigendecomposition, an FFT/DCT spectral reference
for uniform grids, and 50-digit `mpmath` block solutions as independent
references across 200+ synthetic cases: `p4`/`p7`/`p10`, `t*||L||_inf` from
1e-300 to 1e5, one-shot and 16-request sessions, singular and positive
definite operators, signed off-diagonals, isolated and zero rows, zero and
impulse right-hand sides, `n` from 1 to 120000 and `nnz` up to 1e6.  All
passed; the largest observed error was well below the acceptance threshold.
Repeated public-set submissions were also checked through
`/opt/frontieram/bin/submit --test` (all cases passed, no timeouts or
crashes).
