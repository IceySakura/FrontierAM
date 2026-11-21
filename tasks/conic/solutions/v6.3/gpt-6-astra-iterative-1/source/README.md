# Offline Conic Solver

## Build and Run

```sh
./build.sh
./run.sh --input /absolute/input.npz --output /absolute/output.npz
```

The build compiles `ldl.c` and precompiles Python imports into a local cache.
It does not read optimization instances. The run script uses `exec`, writes
only the requested output, disables bytecode writes, and keeps the inherited
CPU affinity. All numerical libraries and HiGHS are restricted to one thread.
There are no network, package installation, agent-tool, or cross-instance
state dependencies.

Installed dependencies: Python 3.12.14, NumPy 2.3.5, SciPy 1.16.3, the HiGHS
1.8.0 engine bundled with SciPy, GCC 12.2.0, and the system C/math libraries.
The HiGHS interface deliberately targets this fixed SciPy distribution.
The native source is included; the build recreates its shared library.

## Methods

- LPs use HiGHS simplex or IPM, with direct array loading and a NumPy-only
  fast entry path. IPM does not require crossover; failed verification can
  trigger crossover or refinement. Bounded auxiliary LPs recover certificates.
- Linearly constrained QPs use primal-dual predictor-corrector iterations,
  with a homogeneous embedding and HiGHS as alternative routes.
- General cones use a homogeneous primal-dual embedding, Nesterov-Todd scaling
  for symmetric cones, and a logarithmic dual barrier with neighborhood
  control for exponential cones.
- Sparse Newton systems use the included reusable LDL factorization where
  appropriate, with iterative refinement and pivoted SuperLU fallback.
  Dense bound QPs use Cholesky and a small equality Schur complement.
- PSD Newton systems exploit sparse coefficient rows or batched congruences.
  Explicit cone-variable formulations can be dualized to reduce dimension.
- Presolve handles fixed variables, constant rows, block-diagonal PSD
  structure, two-dimensional cone transformations, and exposed cone faces.
  All substitutions include primal/dual recovery.
- Large models can use a short linear-relaxation probe for certificates.
  Diagonal, zero-linear-cost QPs also test feasibility in the objective kernel.
- Separable bounds, eligible exponential epigraphs, scalar PSD pencils, and
  eligible single-SOC least-squares problems have direct numerical routes.

Input uses full symmetric CSC `P`, CSC `A`, arbitrary cone ordering, and
upper-triangular, column-major PSD vectors with sqrt(2) off-diagonal scaling.
Optimality and infeasibility certificates are checked in the original
coordinates. Objective constants are not used to relax the gap test.
Optional objective reports are omitted. Unverified results are reported as
`unresolved`, with schema version 1 and finite, correctly shaped vectors for
the other statuses.

## Verification

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python selftest.py
```

This checks PSD encoding, exponential Hessian inverses, native factorization,
duplicate CSC entries, and fresh-process NPZ interfaces for optimal solutions
and both certificate types.

Development also used the five supplied sample problems, six interface
examples, independently checked synthetic optimality/certificate problems,
coefficient rescalings spanning twelve orders of magnitude, and interleaved
backend timing comparisons. All five supplied development cases passed three
fresh-process runs within their published limits. Public feedback was used
to prioritize timeout reduction; it does not establish held-out coverage.

## References

The implementation follows standard ideas from Mehrotra's predictor-corrector
method; Nesterov and Todd's self-scaled cone framework; homogeneous self-dual
conic embeddings; Skajaa-Ye-style nonsymmetric-cone neighborhood control;
sparse elimination-tree LDL factorization; and the HiGHS implementation
distributed with SciPy. The conic iteration, cone algebra, native
factorization, transformations, recovery, and verification code here were
implemented offline. No external solver source or packages were downloaded.
