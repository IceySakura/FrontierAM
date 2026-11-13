# Gravitational Acceleration Solver

Build offline:

```sh
./build.sh
```

Run:

```sh
GALAXY_CPU_LIMIT_S=2 GALAXY_RUN_SEED=0 ./solve INPUT OUTPUT
```

The build requires Linux amd64, an AVX2-capable CPU, and `g++` with C++17
support. There are no downloaded dependencies. Execution uses one process
and one CPU thread, inheriting the caller's CPU affinity.

The input consists of little-endian binary64 rows `(mass, x, y, z)`.
The output contains binary64 `(ax, ay, az)` rows in the original particle
order and coordinate system. The kernel uses `G = 1` and `epsilon = 0.01`.

## Method

- Spatial binary tree, centered on cell centers of mass.
- Mutual dual-tree interactions with Cartesian harmonic multipoles for the
  Newtonian component and Cartesian expansions for the softening correction.
- Full softened-kernel expansions where softening dominates, including
  interactions within a cell.
- Force-relative acceptance based on cell multipole strengths, radial moments,
  and target force-weighted radius moments.
- Adaptive expansion orders and vectorized exact summation for nearby or
  sufficiently small cell pairs.
- Principal-axis alignment for strongly flattened distributions, with inverse
  rotation of the final acceleration vectors.
- Per-input CPU calibration, runtime budget checks, completion reserves, and
  memory-based leaf-size adjustment.

Only the current input, the supplied CPU budget, and fixed mathematical
coefficients are used. No snapshot-dependent data is stored across calls.
