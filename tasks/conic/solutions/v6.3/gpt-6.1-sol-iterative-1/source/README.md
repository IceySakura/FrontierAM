# Sparse Conic Solver

## Build And Run

```sh
./build.sh
./run.sh --input /absolute/input.npz --output /absolute/result.npz
```

The entry point uses `exec` and accepts either order of the two arguments.
Input and output follow the task's schema-version-1 NPZ format. Objective
values are omitted; the checker computes them from the returned vectors.
No network, package installation, agent tools, or writable source directory
is required during solving.

Dependencies are the preinstalled Python 3.12.14, NumPy 2.3.5, SciPy 1.16.3
(including HiGHS 1.8.0 and SuperLU), and GCC/G++ 12.2.0 for the C++17 build.
The build compiles the two supplied C++ files and prepares a local bytecode
cache. All persistent build products stay in this directory. Numerical
libraries are restricted to one thread; inherited CPU affinity is respected.

## Methods

- Exact structural shortcuts include zero optima, diagonal unconstrained
  optima, separable box QPs, and algebraic infeasibility certificates.
- LPs and small QPs use the bundled HiGHS library. Other polyhedral QPs
  use fixed-factor operator splitting, adaptive penalties, and active-set
  KKT polishing, followed by a homogeneous interior-point fallback.
- Singleton bound rows are eliminated analytically. Diagonal QPs use a
  smaller Schur system where appropriate. Work limits depend on sparsity
  rather than wall-clock timing, keeping repeated numerical paths stable.
- The general solver uses a quadratic homogeneous embedding,
  predictor-corrector steps, cone-preserving Ruiz equilibration, and a
  fixed-scale finishing phase. Symmetric cones use Nesterov-Todd scaling.
  The homogeneous scalar Newton denominator is evaluated with nonnegative
  energies to avoid cancellation, and refinement respects residual blocks.
- SOC inverse scaling has a sparse rank-two quasi-definite expansion.
  Exponential blocks use analytic barrier gradients, dual shadow points,
  two secant updates, and factored inverse scaling. Smooth cone faces
  can also receive local nonlinear KKT refinement.
- PSD blocks use the required isometric upper-triangle encoding. Their
  Schur contributions exploit sparse coefficient support or bounded-size
  tensor batches. Structural block-diagonal cones are split exactly.
- Constant linear rows and selected affine equalities are eliminated with
  primal and dual recovery. Aggressive affine elimination is disabled for
  quadratic objectives to avoid worsening their conditioning.
- Sparse ordering and matrix structures are reused. SuperLU is the main
  factorization engine, with a supplied sparse LDL kernel for suitable
  low-fill systems and iterative refinement for regularized solves.

Every accepted solution or certificate is checked in the original
coordinates. Unsuccessful numerical paths return `unresolved`.
All decisions use the current input, not its name or any stored answer.

## Validation And References

Development includes the five supplied sample cases, six interface
examples, generated KKT systems spanning all cones, and generated primal
and dual infeasibility certificates, checked at the specified `1e-8`.
Fresh-process timings include imports and I/O. Larger generated tests
informed bound elimination, sparse ordering reuse, and the quadratic
presolve policy. Public-test feedback is not a held-out score guarantee.

Main algorithmic references: Mehrotra (1992); Nesterov and Todd (1997);
Davis, sparse LDL factorization (2005); Skajaa and Ye (2015);
Huangfu and Hall, HiGHS dual simplex (2018); Stellato et al., OSQP (2020);
Goulart and Chen, Clarabel's quadratic homogeneous embedding (2024).
