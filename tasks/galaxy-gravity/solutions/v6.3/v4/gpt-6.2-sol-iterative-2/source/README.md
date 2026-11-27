# Galaxy acceleration solver

Build from this directory with `./build.sh`. It uses the installed C++17
compiler and FFTW3, and generates FFT plans offline without reading any
snapshots. Run `./solve INPUT OUTPUT` to compute one snapshot.

The input is headerless little-endian binary64 `m, x, y, z` records, with
equal mass per particle. The output contains binary64 `ax, ay, az` records
in input order. Gravity uses `G = 1` and Plummer softening `epsilon = 0.01`.

The main solver uses an adaptive octree and tensor interpolation of the
softened potential. It translates separated cells by small FFT convolutions,
then differentiates the local interpolant. Close pairs use a symmetric
vectorized direct sum. A Cartesian Taylor solver is available when a grid
plan would exceed the budget.

The program is single-process and single-threaded. It chooses a plan using
the current input geometry and `GALAXY_CPU_LIMIT_S`. It does not use
`GALAXY_RUN_SEED` or keep results between calls.
