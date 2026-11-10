#!/bin/sh
# Usage: ./run.sh /abs/input.npy /abs/output.npy
DIR="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export SEED="${SEED:-1}"
exec python3 "$DIR/lrcov.py" "$1" "$2"
