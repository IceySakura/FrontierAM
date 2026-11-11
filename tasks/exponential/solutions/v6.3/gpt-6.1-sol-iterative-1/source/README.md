# Sparse Exponential Propagation

A self-contained C++ implementation of the FAMEXP2 streaming protocol for

```text
y = exp(-t L) b0 + t phi1(-t L) b1.
```

It accepts the supplied symmetric PSD matrix as stored, including singular
matrices, signed off-diagonal entries, potentials, and signed right-hand sides.

## Build and Run

Dependencies are the installed C++17 compiler, Eigen 3 headers, and OpenBLAS.
The build targets Linux amd64 with the AVX2/FMA instruction set available in the
portable-v1 environment. It does not download anything.

```sh
bash build.sh
./run.sh
```

The executable is also included. `run.sh` uses relative paths, inherits CPU
affinity, and sets numerical-library thread counts to one. Python and agent
tools are not runtime dependencies.

The only standard-output bytes are `READY\n` followed by the requested binary
float64 vectors. Each response is written completely before reading the next
request. The process exits normally after the last response. Optional
`FAM_DEBUG=1` diagnostics use standard error.

## Algorithm

- Diagonal and disjoint 2-by-2 blocks use analytic matrix functions. Small
  general matrices and disconnected components of at most 24 vertices use
  extended-precision symmetric eigendecompositions.
- The general fallback is a Chebyshev expansion on a safe spectral interval.
  Gershgorin bounds are tightened by positive-vector bounds when worthwhile.
  Scaled Bessel coefficients use a stable backward ratio recurrence.
  Positive tail sums give the phi1 and phi2 coefficients.
- A shared modified Clenshaw recurrence applies both functions with one sparse
  product per polynomial degree. The truncation budget is 4% of the allowed
  mixed tolerance. Safe lower bounds on output magnitude include decoupled
  rows, spectral contraction bounds, and a mean-conservation bound with an
  explicit defect involving the actual `L * ones`.
- Adaptive symmetric Lanczos trials use integrated-residual and successive
  approximation checks. Expensive trials fall back to the polynomial method.
  Separate initial/source contributions are accepted only after their combined
  error budget is checked.
- A verified diagonal sign similarity handles balanced signed diffusion
  matrices without changing their spectrum or the requested operation.
- A steady-state PCG path is used only if a positive spectral certificate proves
  the omitted transient is below 1% of `atol`. True residuals are recomputed with
  compensated products and checked using a comparison-matrix barrier. Failed
  trials are bounded and disabled for the remainder of the session.
- The p10 path uses compensated vector updates and extended-precision products
  where needed. Near-cancelling inputs use a stable phi-function rearrangement
  with an accurate sparse product, or a full extended-precision recurrence.

Zero time, zero operators, and negligible right-hand sides have direct paths.
No matrix inverse is assumed to exist. Operator preparation and storage are
reused only within the current session; future right-hand sides are never read
ahead. Large buffers may request transparent huge pages, with ordinary-page
fallback if unavailable.

## Validation

The supplied six-case sample manifest was checked with three fresh processes
per case. Additional development checks covered zero/time-zero operators,
16-request lifecycles, positive off-diagonal singular matrices, cancellation,
subnormal entries, high-degree graphs, and generated PSD structure families.
Forty-case p7 and p10 stress sweeps passed. Large p10 tests included matrices
near 120,000 rows and 840,000 stored nonzeros at maximum scaled stiffness.
The final source was rebuilt with a minimal environment and produced a
byte-identical executable. AddressSanitizer and UndefinedBehaviorSanitizer
checks passed on the samples and saved control/regression cases.

In the provided development workspace, the sample check is:

```sh
python /workspace/task/tools/run_local.py \
  --manifest /workspace/task/samples/manifest.json \
  --command "./run.sh" --repeats 3 --output /workspace/check.json
```

The public task files are used only for this development check, not by the
submission at runtime.
