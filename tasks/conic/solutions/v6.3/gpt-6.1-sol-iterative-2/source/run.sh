#!/bin/sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
if [ -x "$(dirname "$0")/native" ]; then
    exec "$(dirname "$0")/native" "$@"
fi
exec python "$(dirname "$0")/main.py" "$@"
