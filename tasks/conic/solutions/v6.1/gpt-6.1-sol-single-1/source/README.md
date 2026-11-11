# Offline Conic Solver

## Running

```sh
./build.sh
./run.sh --input /absolute/input.npz --output /absolute/result.npz
```

The build compiles `kernels.cpp` into `libkernels.so`. It needs only the
installed C++ compiler and standard library. The entry point uses `exec`,
sets numerical libraries to one thread, and preserves inherited CPU affinity.
It does not use networking, package installation, agent tools, writable source
directories, instance names, or cross-invocation state.

## Fixed Dependencies

- Linux amd64, Python 3.12.14
- NumPy 2.3.5, including its bundled ILP64 OpenBLAS/LAPACK
- SciPy 1.16.3, including its installed HiGHS 1.8.0, SuperLU, and sparse kernels
- GCC/G++ 12.2.0 for the instance-independent build

The lightweight paths load the numerical extensions directly, avoiding
unrelated Python package initialization. Their internal ABI is intentionally
pinned to the supplied environment.

## Methods

- LPs use HiGHS, with an artificial-variable phase for Farkas certificates and
  a bounded homogeneous LP for recession certificates. Objectives and extreme
  global coefficient scales are normalized before calling the backend.
- QPs combine native active-set optimization, active-face refinement,
  semismooth dual Newton for diagonal box QPs, dense/sparse Mehrotra
  predictor-corrector steps, and fully corrective conditional gradients for
  low-rank diagonal objectives. Recognizable weighted L1 least-squares
  epigraphs use compiled cyclic coordinate descent, with full conic primal and
  dual reconstruction.
- SOC problems use sparse primal-dual Newton equations in the Lorentz Jordan
  algebra, eliminating nonnegative slack directions. An optional linear
  relaxation supplies warm starts and infeasibility certificates.
- Recognizable exponential epigraphs are eliminated analytically. The
  resulting smooth convex problem uses exact gradients and Hessians with
  primal-dual Newton steps and active-face refinement. General exponential
  blocks also have a barrier-based fallback.
- PSD blocks use Nesterov-Todd scaling. Scaling is computed through Cholesky
  factors and an SVD, avoiding squared-condition-number eigenproblems.
  Schur assembly uses only the matrix entries present in the input operator.
- Supporting-plane outer approximation provides an additional conic fallback.

Affine equality elimination and equilibration are reversible. Solutions and
certificates are restored to original variables and cone order. Output claims
are checked against the original residual, cone, gap, and certificate
conditions; unsuccessful paths write `unresolved`. Objective constants are
not reported, so no transformed objective offset enters an acceptance test.
Certificate matrices are normalized before multiplication, including for
subnormal input scales. Duplicate sparse entries are summed before solving.

## References and Development

The implementation was developed offline from the installed numerical
interfaces and the mathematical contract. Principal algorithmic references:

- S. Mehrotra, "On the Implementation of a Primal-Dual Interior Point Method,"
  SIAM Journal on Optimization, 1992.
- Y. Nesterov and M. Todd, self-scaled barriers and primal-dual interior-point
  methods, Mathematics of Operations Research, 1997 and 1998.
- S. Boyd and L. Vandenberghe, *Convex Optimization*, 2004, including conic
  duality, logarithmic barriers, and geometric programming.
- Q. Huangfu and J. A. J. Hall, dual revised simplex implementation, 2018;
  the installed HiGHS implementation supplies the LP/QP backend.
- The installed SciPy sparse-kernel and SuperLU interfaces, and NumPy LAPACK.

Development uses the supplied interface examples and public NPZ inputs.
Independent checks are run with the supplied `check_result.py`, outside the
submission. Experiments motivated lightweight imports for short deadlines,
epigraph condensation for large exponential models, and SVD-based SDP
scaling for accurate boundary iterates. No development input or stored answer
is included in this submission.

The public interface examples and synthetic empty-dimension, duplicate-entry,
mixed-cone, and extreme-scale checks are used for interface validation.
Difficult or weakly conditioned problems can still return `unresolved`; the
solver does not claim universal convergence within every case's deadline.
