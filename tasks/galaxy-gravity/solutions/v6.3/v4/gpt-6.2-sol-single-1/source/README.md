# Softened gravitational acceleration

Build and run on Linux x86-64 with AVX2 and FMA:

```sh
./build.sh
./solve INPUT OUTPUT
```

No external libraries, generator, or network access are required. The solver
accepts the headerless little-endian binary64 format from `problem.md` and
writes the acceleration vectors in the input particle order.

The implementation partitions the input into a spatial binary tree. A mutual
tree walk assigns pairs either to direct summation or to high-order Taylor
translations. Distant translations use a compressed harmonic expansion plus
an explicit softened-kernel correction. Both corrections and the direct
interactions use the prescribed Plummer softening length of 0.01. An
optional final pass spends remaining CPU time on the particles that are most
sensitive to the approximation.

The solver uses one thread and only the current input. It uses
`GALAXY_CPU_LIMIT_S` to select the expansion tolerance and checks process CPU
time while computing. The coefficient storage is capped, and the harmonic
and full Cartesian stages do not hold their large arrays concurrently.
`GAL_DEBUG=1` reports local timings and peak resident memory on standard error.
