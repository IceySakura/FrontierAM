# SLEPc MFN panel member

This is the HUMAN-SOTA SLEPc baseline for FAMEXP2. It uses the unmodified
SLEPc 3.26.1 `MFNEXPOKIT` matrix-function solver with `FNEXP`, built on
unmodified PETSc 3.26.1 configured with MPIUNI. Both projects use the
2-clause BSD license; their notices are included.

## Run

```sh
./run.sh [tolerance-multiplier]
```

The optional multiplier is used only for public calibration. The default is
`100`, so the SLEPc tolerance is `100 * rtol`. `run.sh` forces common numerical
thread pools to one thread. The executable has an inherited
`$ORIGIN/lib` ELF RPATH, so the directory can be copied anywhere and needs no
PETSc or SLEPc installation.

## Adapter

For each session the adapter builds the `(n+1)`-square sparse matrix

```text
A = [ -t L   t b1 ]
    [   0       0 ]
```

and computes `exp(A) [b0; 1]` with SLEPc MFN. The first `n` entries are the
requested result. The CSR-derived block is assembled once. For each request
only the preallocated final column and input vector change; the same `Mat`,
`MFN`, and vectors are reused for all 16 requests. The documented Expokit
Krylov dimension is fixed at 60 for one-shot sessions and 80 for reuse
sessions. No future right-hand side is read before the current answer is
written.

`src/stream_slepc.c` is only a FAMEXP2/CSR adapter and parameter policy. PETSc
and SLEPc solver sources were not changed.

## Packaged build

The shipped amd64 libraries were built inside
`frontieram-exponential:port-v1` (Debian bookworm, GCC 12.2) on CPUs 2 and 3:

```text
PETSc:
  --with-mpi=0 --with-debugging=0 --with-shared-libraries=1
  --with-cxx=0 --with-fc=0 --with-fortran-bindings=0
  --with-x=0 --with-ssl=0 --with-cuda=0 --with-hip=0
  --with-kokkos=0
  --with-blaslapack-lib=/usr/lib/x86_64-linux-gnu/libopenblas.so
  COPTFLAGS="-O3 -march=x86-64-v2" CFLAGS="-fPIC"
SLEPc:
  PETSC_DIR=<installed PETSc> ./configure --prefix=<install>
```

The build used no downloaded optional packages. Runtime files were stripped.
No `build.sh` is supplied because a clean PETSc build took longer than the
evaluator's optional 600-second build allowance; the ready-to-run binary and
all non-system shared libraries it needs are included instead.

Method references:

- V. Hernandez, J. E. Roman, and V. Vidal, “SLEPc: A Scalable and Flexible
  Toolkit for the Solution of Eigenvalue Problems,” ACM TOMS 31(3), 2005.
- SLEPc Users Manual, Matrix Function (MFN) chapter.
