#!/bin/sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$DIR/run_lrcov.py" "$1" "$2"
