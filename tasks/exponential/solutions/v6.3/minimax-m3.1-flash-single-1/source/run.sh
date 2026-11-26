#!/bin/sh
# FAMEXP2 entry point.  Single-threaded BLAS keeps the CPU budget honest.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
       PYTHONUNBUFFERED=1
exec python3 "$(dirname "$0")/solve.py"
