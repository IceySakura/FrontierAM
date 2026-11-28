# Exponential propagation solver

This submission implements the FAMEXP2 binary session protocol in C++.

## Algorithm

The program computes

`y = exp(-t L) b0 + t phi1(-t L) b1`

for a real symmetric positive semidefinite sparse CSR matrix `L`.

Paths are selected from the actual input only:

* `t = 0`, zero operators, and diagonal operators are evaluated elementwise.
* Operators whose connected components have size at most two are evaluated
  analytically with stable 1x1 and 2x2 divided differences.
* Small dense operators (`n <= 256`) use a cached symmetric eigendecomposition.
* Large operators use a Chebyshev expansion on a safe spectral interval.
  The upper bound combines the Gershgorin bound with a short fully
  reorthogonalized Lanczos run.  A positive Gershgorin lower bound is used when
  it is numerically safe; if the exponential term is then below the requested
  tolerance, it is omitted.  The degree is chosen from the tolerance and, for
  shifted intervals, from the Bernstein-ellipse decay rate.  Evaluation stops
  when both a coefficient-tail estimate and a recent partial-sum difference are
  small relative to the mixed absolute/relative tolerance.
* When both right-hand sides are active, sparse products are fused so the CSR
  matrix is read once for both vectors.  For large degree-heterogeneous
  operators, a symmetric degree-ordering permutation is applied once to improve
  sparse-product cache behavior; vectors are permuted and scattered back.
* Very small `eta` uses a fused truncated Taylor series and skips Chebyshev
  preparation.

All preparation is performed after the session header arrives.  No state is
shared between processes or sessions.

## Dependencies

Only the frozen offline environment is used:

* g++ with C++17
* LAPACK/LAPACKE (`liblapacke`, `liblapack`)
* OpenBLAS (`libopenblas`)

No network access or package installation is required.

## Build and run

```sh
bash build.sh
./run.sh
```

`run.sh` forces single-threaded BLAS and executes the solver.  If the binary is
missing, it invokes `build.sh` as a fallback.

## Public checks

The supplied sample manifest can be checked with the task runner:

```sh
python /workspace/task/tools/run_local.py \
  --manifest /workspace/task/samples/manifest.json \
  --command ./run.sh --repeats 3 --output dev_report.json
```

Note: the supplied sample manifest states that scored sample reference arrays
are placeholders; controls are qualified.  Development validation therefore also
uses independently recomputed dense-eigenvector references for the small
sample matrices, plus larger synthetic CSR cases checked against high-degree
Chebyshev references.
