# Gravitational Acceleration Solver

Build offline with `./build.sh`, then run `./solve INPUT OUTPUT`.

The solver is a single-threaded C++17 program with no external dependencies.
It directly sums all particle interactions, using AVX2/FMA and symmetric
cache-blocked interactions for the equal-mass snapshots. The softened kernel
is used for every pair. A fourth-order binary64 correction brings the SIMD
reciprocal-square-root estimate to double-precision accuracy.

Input and output use the specified headerless little-endian binary64 formats.
