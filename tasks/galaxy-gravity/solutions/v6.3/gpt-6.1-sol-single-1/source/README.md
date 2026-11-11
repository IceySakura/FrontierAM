# Gravitational Acceleration Solver

Build offline with `./build.sh`, then run:

```sh
./solve INPUT OUTPUT
```

The implementation is single-threaded C++17 and uses only the system C++
standard library and AMD64 AVX2/FMA instructions. The build inherits the
machine's CPU affinity and does not download anything.

The solver combines adaptive Cartesian and harmonic multipole expansions,
exact softened near-field interactions, one-sided evaluation for selected
asymmetric interactions, and shared translation operators on a spatial
hierarchy. The Plummer softening correction is included in far-field
interactions as well as direct interactions. Expansion orders are chosen
per interaction, with a process-CPU-time governor for the final workload.

Input and output are headerless little-endian binary64 arrays in the
specified particle order. All preprocessing and translation caches are
local to the current process; no snapshot information persists between
invocations.
