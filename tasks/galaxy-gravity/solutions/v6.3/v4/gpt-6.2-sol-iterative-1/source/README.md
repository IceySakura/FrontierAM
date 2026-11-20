# Galaxy accelerations

Build offline with `./build.sh` and run `./solve INPUT OUTPUT`. The
implementation uses one process and one thread. It needs the system C++
compiler and standard runtime only; no FFT libraries or downloads are needed.

The solver uses a spatial tree. It sums close interactions directly with the
Plummer force and uses variable-order Taylor expansions of the same softened
potential for separated cells. For some distant interactions it evaluates the
harmonic part and the softening correction separately. It uses
`GALAXY_CPU_LIMIT_S` to set the expansion tolerance, with a reserve for
evaluation and output. All planning and data are local to each invocation.
