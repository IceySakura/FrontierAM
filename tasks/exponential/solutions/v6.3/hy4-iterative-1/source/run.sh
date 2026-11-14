#!/bin/sh
# FAMEXP2 entry point.  Runs from the submitted directory.
cd "$(dirname "$0")" || exit 1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
if [ ! -f libfamexp.so ]; then
    sh build.sh >&2
fi
exec python3 -u solver.py
