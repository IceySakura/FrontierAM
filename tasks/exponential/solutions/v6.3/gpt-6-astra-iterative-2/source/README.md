# Sparse Exponential Propagation

This submission implements FAMEXP2 and computes
`exp(-t L) b0 + t phi1(-t L) b1`, including singular PSD matrices,
signed entries, zero time, and sequential 16-request sessions.

## Build and Run

```sh
bash build.sh
./run.sh
```

The build uses the installed GNU C++17 compiler, LAPACKE/OpenBLAS headers,
and these installed libraries: LAPACKE, OpenBLAS, SuiteSparse UMFPACK
(`libumfpack.so.5`), and FFTW3 (`libfftw3.so.3`). Their dependencies, including
METIS, are supplied by the frozen numerical environment. No downloads,
package installation, Python runtime, agent tools, or external files are
needed by the solver. The included binary is reproducible with `build.sh`.

The target is Linux amd64 with AVX2/FMA. Execution uses one thread and
inherits CPU affinity; no CPU index is hard-coded. Generic library
initialization and pipe configuration precede `READY`. All input-dependent
analysis, plans, factorizations, and numerical work follow it.

## Algorithms

* Scalars and independent 2-by-2 blocks use analytic formulas, stable
  `expm1`, and a compensated determinant. Small disconnected components
  use extended-precision Jacobi eigendecomposition.
* Uniform Cartesian operators use FFT, sine, or cosine transforms only
  after every stencil value, boundary diagonal, and sign relation is
  checked against the input matrix.
* Low stiffness uses a Taylor recurrence with an infinity-norm remainder
  bound. Negligible corrections can be omitted within the requested
  tolerance.
* The general fallback is a combined Chebyshev/Clenshaw expansion.
  Gershgorin and positive-vector bounds enclose the spectrum. Scaled
  Bessel ratios produce exponential coefficients; weighted coefficient
  tails produce phi1 coefficients. Both right-hand sides share one
  sparse multiplication per degree.
* Selected stiff operators use Lanczos with a tridiagonal resolvent error
  bound. Diagonal dominance permits a tighter infinity-norm bound.
  A cancellation guard protects the combined phi identity.
* Reuse sessions may cache eight complex shifted sparse factorizations
  for the order-16 Chebyshev rational approximation. Symbolic fill,
  memory estimates, and the measured first solve decide whether this
  preparation is worthwhile. Important poles receive refinement at p10.
* Checked null-space shortcuts and an extended-precision polynomial
  fallback handle delicate cancellation.

State is retained only within one process/session. Each response is written
completely before the next right-hand side is read. The process exits with
status zero immediately after its final synchronous write.

## Checks

`python check.py` runs self-contained numerical and streaming checks using
the installed NumPy/SciPy. These are independent of the solver and require
no task data. They do not emulate evaluator isolation or scoring.

The supplied public samples were also checked in three fresh processes:

```sh
python /workspace/task/tools/run_local.py \
  --manifest /workspace/task/samples/manifest.json \
  --command "./run.sh" --repeats 3 --output /workspace/dev_report.json
```

Development checks additionally covered 100,000-row signed grids, general
sparse operators across all accuracy levels, 16-request reuse, and
near-cancelling right-hand sides. No sample matrices or answers are embedded
in the solver. `PROP_DIAGNOSTIC=1` enables optional stderr diagnostics.
