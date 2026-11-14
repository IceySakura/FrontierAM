#!/bin/sh
# Offline conic solver entry point.
set -e
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
exec python3 -B "$DIR/solver.py" "$@"
