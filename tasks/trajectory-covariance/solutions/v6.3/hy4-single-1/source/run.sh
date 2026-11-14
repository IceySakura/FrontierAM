#!/bin/sh
# Usage: ./run.sh /absolute/input/input.npy /absolute/output/pred.npy
set -eu
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$DIR/estimate.py" "$1" "$2"
