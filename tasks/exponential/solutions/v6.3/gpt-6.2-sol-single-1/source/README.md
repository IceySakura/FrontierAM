# Exponential propagation (FAMEXP2)

Build with `./build.sh` and launch with `./run.sh` from this directory.
The submitted executable can also be run directly through `./run.sh`.
The offline build uses GCC's C++17 toolchain and targets Linux amd64 with
AVX2 and FMA. The C/C++ runtime is required. If the installed
`libfftw3l.so.3` is present, the solver can use it for verified tensor stencils;
the rest of the solver does not depend on FFTW. There are no downloads.

`run.sh` serves one FAMEXP2 session. It emits `READY`, reads one CSR matrix,
then reads each right-hand-side pair only after sending the last full result.
The solver is single-threaded, inherits its CPU affinity, and has no cache
outside the process.

## Numerical methods

- A centered Taylor series uses a matrix infinity-norm bound in short sessions.
  Diagonal and independent two-by-two blocks have direct scalar formulas.
- The general method uses a Gershgorin interval, with optional diagonal
  similarity to tighten the bounds. On that interval, both matrix functions
  have a Chebyshev expansion. The exponential coefficients come from backward
  Bessel ratios and normalization; the integral coefficients follow from the
  corresponding three-term coefficient equation.
- A shared, endpoint-stabilized Clenshaw recurrence combines the coefficient
  contributions of both right-hand sides. It uses one sparse matrix-vector
  product per degree, with stencil or four-row SIMD layouts where profitable.
  Truncation uses the coefficient tails and the input vector norms. Sensitive
  outputs use a mixed binary64/compensated recurrence or an extended-precision
  scalar recurrence.
- Small connected components and exact separable stencils have direct spectral
  paths. Short Krylov, scalar-mode, and steady-state trials can return early
  only when their residual estimates are within the output tolerance budget.

## Public checks

From `/workspace/task`:

```sh
python tools/run_local.py --manifest samples/manifest.json \
  --command /workspace/submission/run.sh --repeats 3 --output /workspace/public_report.json
```

The six public cases passed three repetitions. Additional local checks used
independent long-double forward Chebyshev references, analytical block and
hub spectra, and Fourier/cosine transforms for regular and permuted grids.
They exercised all three tolerances, cancellation, and sequential 16-request
sessions near the matrix-size limit. Local checks do not produce a hidden
score.
