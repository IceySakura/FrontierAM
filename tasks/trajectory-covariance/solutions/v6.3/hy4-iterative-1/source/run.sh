#!/bin/sh
# Estimate the long-run covariance of a trajectory.
# Usage: ./run.sh /absolute/input/input.npy /absolute/output/pred.npy
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
exec python3 "$HERE/lrc.py" "$1" "$2"
