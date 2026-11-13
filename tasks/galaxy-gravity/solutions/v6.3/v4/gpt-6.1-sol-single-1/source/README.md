# Gravitational Acceleration Solver

Build offline with `./build.sh`, then run `./solve INPUT OUTPUT`.
The build uses the installed C++17 compiler and Python standard library only.
The executable has no runtime package dependencies.

The solver uses a spatial binary tree, SIMD softened direct interactions,
Cartesian harmonic multipoles for the long-range field, and a Cartesian
expansion of the Plummer softening correction. Close cell pairs use the
softened kernel without the harmonic split. Sparse outer targets are
evaluated with an independent high-order tree traversal.

Runtime kernel measurements and interaction counts select opening criteria
from `GALAXY_CPU_LIMIT_S`. All computations are single-threaded. No state is
retained between calls. `GALAXY_RUN_SEED` does not affect this deterministic
implementation.
