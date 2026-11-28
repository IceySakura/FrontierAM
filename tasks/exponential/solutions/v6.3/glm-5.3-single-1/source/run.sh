#!/bin/sh
# FAMEXP2 entry point.  Generic imports, kernel compilation and all input
# independent initialisation happen before READY and are therefore untimed.
cd "$(dirname "$0")" || exit 1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec python3 -B solver.py
