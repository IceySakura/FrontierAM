#!/bin/sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
base="$(dirname "$0")"
if [ ! -f "$base/libpcubature.so" ]; then
    cc -O2 -fPIC -shared "$base/vendor/cubature/pcubature.c" -lm -o "$base/libpcubature.so"
fi
exec python3 "$base/solver.py"
