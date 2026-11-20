#!/bin/sh
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
base=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 -B "$base/estimate.py" "$@"
