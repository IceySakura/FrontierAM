# Sparse Exponential Propagation

An offline, single-threaded C++17 implementation of FAMEXP2, computing

    exp(-t L) b0 + t phi1(-t L) b1.

## Build And Run

```
./build.sh
./run.sh
```

`build.sh` compiles the complete implementation in `solver.cpp`. `run.sh` also
builds if the executable is absent. It works from a different working directory
and uses no files outside this submission except installed system dependencies.

Dependencies: GCC with C++17 support, Eigen3 headers at
`/usr/include/eigen3`, and the standard C++/math libraries. The supplied amd64
environment supports the AVX2/FMA instructions selected by `-march=x86-64-v3`.
No Python, agent tools, network access, downloads, or runtime JIT are required.
The process inherits its assigned CPU affinity; no CPU index is assumed.

## Algorithm

- Zero time and zero matrices have direct answers.
- At low stiffness, an inhomogeneous Taylor recurrence uses an adaptive
  infinity-norm remainder bound.
- Disconnected blocks of order at most 24 use extended-precision spectral
  evaluation. Two-by-two blocks have a cancellation-safe analytic eigensystem.
- The general path bounds the PSD spectrum with Gershgorin and positive
  diagonal similarities. Scaled modified-Bessel coefficients are computed by
  backward ratios and normalization. Positive suffix sums give the phi1
  coefficients without division by the matrix.
- Both functions are evaluated together in one endpoint-stabilized Clenshaw
  recurrence. Truncation is bounded using coefficient tails and RHS two-norms.
  A residual-controlled linear functional can certify a larger output scale.
- Sparse row compensation improves diffusion/nullspace accuracy. An
  input-dependent choice of ordering, CSR, and four-row SIMD storage reduces
  sparse-product cost; excessive padding is avoided.
- Delicate p10 results use the original received binary64 matrix. A
  double-double sparse residual test first checks whether the linear-in-time
  solution is already accurate enough. Otherwise, an extended-precision
  Clenshaw evaluation is used.

Preparation, coefficients, matrix layouts, and work buffers are reused only
within the current session. RHS vectors are read and answered sequentially.
Singularity, positive off-diagonals, nonzero row sums, and signed vectors are
supported. Stdout contains only `READY\n` and the prescribed vector bytes.

## Validation

The installed public runner passed all three fresh repetitions of all
56 `dev` cases and all 40 `dev_full` cases. These are correctness and local
cost checks, not a calibrated score.

From this directory, the checks are reproducible in the development environment:

```
python /workspace/task/tools/run_local.py --manifest /workspace/task/dev/manifest.json --command ./run.sh --repeats 3 --output dev_report.json
python /workspace/task/tools/run_local.py --manifest /workspace/task/dev_full/manifest.json --command ./run.sh --repeats 3 --output dev_full_report.json
```

Additional development checks covered Fourier-solvable periodic diffusion,
exact rank-one PSD matrices, singular integer Gram matrices independently
checked with high-precision arithmetic, zero/cancelling outputs, and diagonal
congruences. Formal-size 16-request checks included a 100,000-variable,
900,000-nonzero periodic operator with `t * norm_inf(L) = 90000`, and a
120,000-variable singular congruence operator. No private tests or reference
timings were accessed.
