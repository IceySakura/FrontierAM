#!/bin/sh
# Offline conic solver entry point.  All heavy numerical work is single
# threaded: the harness pins us to one CPU.
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$DIR" || exit 1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
exec python -B "$DIR/solver.py" "$@"
