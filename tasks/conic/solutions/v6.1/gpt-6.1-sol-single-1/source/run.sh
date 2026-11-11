#!/bin/sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPYCACHEPREFIX="$HERE/.pycache"
exec python "$HERE/fast.py" "$@"
