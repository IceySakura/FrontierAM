#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="$HERE/pycache"
unset PYTHONDONTWRITEBYTECODE
gcc -O3 -std=c11 -fPIC -shared "$HERE/numerics.c" -o "$HERE/numerics.so" -lm
python "$HERE/precompile.py"
