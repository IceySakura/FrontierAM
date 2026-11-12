#!/bin/sh
set -eu
if [ "$#" -ne 2 ]; then
    printf '%s\n' 'Usage: ./run.sh INPUT.npy OUTPUT.npy' >&2
    exit 2
fi
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export SEED=1
exec python3 -B "$(dirname "$0")/estimator.py" "$1" "$2"
