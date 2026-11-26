#!/bin/sh
set -eu
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
if [ ! -f fame_core.so ]; then
    g++ -O3 -march=native -funroll-loops -shared -fPIC fame_core.cpp -o fame_core.so
fi
exec python3 solve.py
