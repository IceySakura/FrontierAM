#!/bin/sh
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export EIGEN_DONT_PARALLELIZE=1
# Optional first argument: the stopping multiplier m (public calibration only); 0.1 is the analytic default.
exec "$HERE/rational" "${1:-0.1}"
