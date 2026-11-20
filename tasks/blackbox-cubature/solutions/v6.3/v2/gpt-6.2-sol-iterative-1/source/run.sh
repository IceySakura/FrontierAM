#!/bin/sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
exec python3 -B "$(dirname "$0")/solver.py"
