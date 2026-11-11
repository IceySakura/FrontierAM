# Gravitational Acceleration Solver

Build offline with `./build.sh`, then run `./solve INPUT OUTPUT`.

The source requires a C++17 compiler and Linux amd64 with AVX2/FMA.
There are no package downloads or additional runtime dependencies.
Input and output use the raw little-endian binary64 layouts in the problem.

The solver combines symmetric direct interactions, adaptive harmonic multipole
expansions, and a Cartesian correction retaining Plummer softening at every
distance. Per-call planning controls accuracy within the CPU budget; a finishing
pass uses remaining CPU time for selected complete direct sums.
