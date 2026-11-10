#!/bin/sh
# Build inside the frozen task image. The Expokit sources and the BLAS/LAPACK subset shipped with them are unmodified.
set -eu
cd "$(dirname "$0")"
build() {
    out=$1; shift
    # Fortran 77 passes workspace arrays across types; gfortran >= 10 needs this to accept it.
    for source in expokit blas lapack; do
        gfortran "$@" -fallow-argument-mismatch -w -c "vendor_expokit/$source.f" -o "/tmp/$source.o"
    done
    gcc "$@" -std=c99 -D_POSIX_C_SOURCE=200809L -c src/stream_expokit.c -o /tmp/stream_expokit.o
    gfortran "$@" -static -o "$out" /tmp/stream_expokit.o /tmp/expokit.o /tmp/lapack.o /tmp/blas.o
}
build expokit_avx2 -O3 -march=x86-64-v3
build expokit_generic -O2
