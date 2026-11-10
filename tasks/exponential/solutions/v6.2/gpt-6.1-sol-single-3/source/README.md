# Exponential Propagation

## Build And Run

```sh
bash build.sh
./run.sh
```

`run.sh` executes the included native binary. `build.sh` reproduces that binary
offline from `solver.cpp` using the installed GCC C++17 compiler and Eigen 3
headers. Runtime dependencies are the standard C++ and math libraries only.
There are no downloads, Python runtime dependencies, agent tools, generated
input-dependent code, or external services.

The executable implements FAMEXP2 directly. It writes and flushes `READY\n`
before reading session input, then returns each complete little-endian binary64
vector before reading the next request. Diagnostics are confined to stderr.
It inherits CPU affinity and uses one thread.

## Algorithm

- A simultaneous Chebyshev expansion evaluates both functions in one sparse
  recurrence. Scaled modified-Bessel coefficients are computed
  by a downward recurrence. Positive weighted tail sums supply the phi
  coefficients, including the value at zero without matrix inversion.
- Absolute row sums and positive-vector Collatz-Wielandt bounds enclose the
  spectrum. Coefficient-tail estimates use the current RHS norms and the global
  output tolerance; an a posteriori check can increase the degree.
- Reinsch's modified Clenshaw recurrence improves behavior near zero
  eigenvalues. Diagonally dominant operators use algebraic edge differences
  while retaining the actual received diagonal residual.
- Runtime-dispatched AVX2/FMA sliced-ELL kernels, bounded row-degree sorting,
  and conditional reverse Cuthill-McKee ordering improve sparse multiplication.
  A scalar CSR path supports CPUs without AVX2/FMA and unusually long rows.
- Extended precision is used for cancellation-sensitive results, stiff
  non-diagonally-dominant p10 cases, and ill-conditioned small spectral blocks.
  Long extended-precision dot products use compensated summation.
- Zero operators, zero time, diagonal matrices, two-node blocks, small
  disconnected components and exactly verified integer eigenvectors have direct
  paths. A certified positive spectral lower bound can eliminate a transient
  smaller than the absolute tolerance and use a reciprocal Chebyshev series.
- At p4/p7, a bounded short Lanczos probe can finish low-dimensional spectra.
  Integrated resolvent bounds and explicit recurrence-roundoff estimates
  determine acceptance; otherwise the solver uses the Chebyshev path.

All matrix-dependent preparation and cached coefficients are session-local.
Future right-hand sides are neither read nor assumed in advance.

## Checks

`python3 selftest.py` runs nine analytic protocol sessions using only Python's
standard library, including 16-request lifecycles, singular and signed matrices,
positive off-diagonal entries, zero time, a stiff scalar and underflow-scale data.

Final native-executable checks:

| Suite | Sessions | Repetitions | Passing Runs |
| --- | ---: | ---: | ---: |
| Public `dev` | 56 | 3 | 168/168 |
| Public `dev_full` | 40 | 3 | 120/120 |
| Independent FFT, analytic, and 65-digit dense references | 14 | 3 | 42/42 |

The largest scaled error across these 330 runs was 0.01089, where 1 is the
acceptance threshold. Two additional maximum-stiffness manufactured p10 reuse
sessions, including a 120,000-row operator and a near-million-entry operator,
passed in approximately 70 and 151 seconds against the 240-second limit.

AddressSanitizer and UndefinedBehaviorSanitizer checks passed all 56 smoke
sessions, six additional targeted sessions, and the standalone self-test, with
no findings. Additional dense subnormal-scale cases also passed.

A fresh source-only directory rebuilt successfully and passed the standalone
self-test. Its executable was byte-identical to the included executable.

All timings are development observations, not calibrated panel comparisons.
No score is inferred, and no private cases or reference timings were accessed.
