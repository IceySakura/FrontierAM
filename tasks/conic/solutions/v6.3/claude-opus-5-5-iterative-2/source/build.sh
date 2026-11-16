#!/bin/sh
# Builds the native solver (single-threaded C99; links the system reference LAPACK/BLAS).
# A prebuilt bin/solver is shipped; it is kept if compilation is not possible.
cd "$(dirname "$0")" || exit 1
CC=${CC:-gcc}
LIBDIR=/usr/lib/x86_64-linux-gnu
LAPACK=$LIBDIR/liblapack.so.3
BLAS=$LIBDIR/libblas.so.3
[ -e "$LAPACK" ] || LAPACK=$(ls $LIBDIR/lapack/liblapack.so.3* 2>/dev/null | head -n1)
[ -e "$BLAS" ] || BLAS=$(ls $LIBDIR/blas/libblas.so.3* 2>/dev/null | head -n1)
mkdir -p bin
if $CC -O3 -std=gnu99 -fno-math-errno -o bin/solver.tmp src/solver.c src/ldl.c src/npzio.c \
       "$LAPACK" "$BLAS" -ldl -lm 2> bin/build.log; then
    mv -f bin/solver.tmp bin/solver
    echo "build ok"
else
    rm -f bin/solver.tmp
    cat bin/build.log
    if [ -x bin/solver ]; then echo "compile failed; using shipped prebuilt bin/solver"; exit 0; fi
    exit 1
fi
exit 0
