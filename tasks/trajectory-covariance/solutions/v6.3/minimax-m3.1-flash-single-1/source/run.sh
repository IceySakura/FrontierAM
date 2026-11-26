#!/bin/sh
# Usage: ./run.sh /absolute/path/input.npy /absolute/path/output.npy
# Estimates the long-run covariance matrix of the trajectory in INPUT and
# writes it (d x d, float64, C order) to OUTPUT.
set -e
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec python3 "$DIR/main.py" "$@"
