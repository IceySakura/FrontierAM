#!/bin/sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export SEED=1
DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$DIR/lrc.py" "$1" "$2"
