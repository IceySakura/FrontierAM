# Gravitational Acceleration Solver

Build offline with `./build.sh`, then run:

```sh
./solve INPUT OUTPUT
```

The input and output use the binary64 layouts specified in the problem.
The implementation is single-threaded C++17 for the supplied Linux amd64
environment. Building requires only the installed Python 3 standard library
and GCC; running requires no Python, packages, auxiliary data, or agent tools.

The solver uses a spatial binary tree, mutual harmonic multipole expansions
for the Newtonian kernel, and Cartesian expansions of the difference between
the softened and Newtonian kernels. Nearby interactions are summed directly
with the specified softening. Expansion centers are centers of mass.

A force-relative opening criterion and a cost model calibrated on the current
input select the working accuracy. Timing guards can replace remaining work
with lower-order complete interactions. No snapshot-dependent state is stored
or read between invocations.

`generate.py` and `harmonic_gen.py` generate the fixed-order kernels included
by `solver.cpp`. The default build uses harmonic order 16 and correction
order 10.
