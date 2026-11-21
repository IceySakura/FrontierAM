# Gravitational Acceleration Solver

Build offline:

```sh
./build.sh
```

Run:

```sh
GALAXY_CPU_LIMIT_S=2.4 GALAXY_RUN_SEED=0 ./solve INPUT OUTPUT
```

The build uses the installed C++17 compiler and Python's standard library.
The executable has no Python or third-party runtime dependencies.

The solver uses an adaptive spatial tree and tenth-order Cartesian
fast-multipole expansions of the Plummer-softened kernel itself. Near
interactions are summed directly in double precision. Sparse outer leaves
are refined geometrically, and strongly flattened distributions are aligned
with their principal axes. Output vectors are transformed back to the input
coordinate system and restored to the original particle order.

Each invocation calibrates its interaction kernels and chooses an opening
tolerance from its CPU budget and the current snapshot's interaction counts.
Interaction buffers are bounded. A final time-bounded particle-to-tree pass
refines particles with large estimated errors. Deadline safeguards preserve
finite complete output. Small inputs use direct summation.

The program is single-threaded and uses the inherited CPU affinity. It reads
only the current input and writes only the requested output. There is no
cross-invocation cache or dependence on the run seed.

Source files:

- `main.cpp`: input/output, tree construction, expansions, scheduling.
- `gen.py`: generates the fixed softened-kernel derivative recurrence.
- `derivatives.h`: generated recurrence, also included for inspection.
- `build.sh`: offline build entry point.
