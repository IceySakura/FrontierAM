#!/bin/sh
# Builds the native solver into ./bin (falls back to the prebuilt binary if compilation fails).
cd "$(dirname "$0")" || exit 1
mkdir -p bin
SRC="src/solver.c src/npzio.c src/ldl.c src/amd.c"
LIBS="-l:libz.so.1 -l:liblapack.so.3 -l:libblas.so.3 -lm -ldl"
if gcc -O3 -std=gnu99 -o bin/conic_solver.new $SRC $LIBS 2> bin/build.log; then
    mv -f bin/conic_solver.new bin/conic_solver
elif [ -x bin/conic_solver.prebuilt ]; then
    cp -f bin/conic_solver.prebuilt bin/conic_solver
fi
[ -x bin/conic_solver ] || exit 1
exit 0
