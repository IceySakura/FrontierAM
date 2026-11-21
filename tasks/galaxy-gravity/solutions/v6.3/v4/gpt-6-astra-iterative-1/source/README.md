# Gravitational acceleration solver

Build offline with `./build.sh`, then run:

```sh
./solve INPUT OUTPUT
```

The build uses Python's standard library to generate fixed-order kernels and
compiles the C++17 solver with AVX2 and FMA enabled. There are no downloaded
dependencies. The executable is single-threaded and leaves CPU affinity alone.

Input consists of little-endian doubles `(mass, x, y, z)`. Output consists of
little-endian doubles `(ax, ay, az)` in the original particle order. The solver
uses the specified equal particle masses, `G = 1`, and `epsilon = 0.01`.

## Numerical method

The spatial hierarchy is an adaptive midpoint-split binary tree. Cells store
centered Cartesian moments and compressed symmetric trace-free moments.
Well-separated cell pairs use mutual multipole-to-local translations. Nearby
pairs, and small pairs for which translation would be more expensive, use
direct softened force sums.

The high-order expansion is not an unsoftened approximation. For squared
separation `r2`, put `s = r2 + epsilon^2`. The trace-free derivative tensor of
rank `n` is the Laplace derivative tensor multiplied by
`(r2 / s)^(n + 1/2)`. A recurrence computes it without division by `r2`, so
coincident expansion centers are safe. A Cartesian residual supplies the
remaining traces through order 10, or order 12 on large snapshots; trace-free
terms extend through order 16.
All near-field interactions use the same softening model.

Expansion evaluation and force sums process four double-precision lanes at a
time. Tree order and optional principal-axis coordinates are undone before
writing the output.

## Resource control

Each invocation reads `GALAXY_CPU_LIMIT_S`, measures representative kernels,
and chooses its separation threshold for that invocation. Near-field work is
completed before far-field translations. The runtime reserves time for local
expansion evaluation and output; lower-order far-field kernels are available
when measured work exceeds the estimate. The emergency near-field kernel
still sums every pair, using a Newton-refined reciprocal square root.
Spare CPU time is used for direct recomputation of particles with the largest
estimated relative errors, with a separate deadline reserved for output.

Moment storage and interaction-list growth have separate memory limits.
No snapshot data, calibration, or results are retained between invocations.
