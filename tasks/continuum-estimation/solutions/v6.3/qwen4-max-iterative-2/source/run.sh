#!/bin/sh
set -eu
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export LC_ALL=C
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -x /workspace/runtime/bin/python3 ]; then
  PY=/workspace/runtime/bin/python3
else
  PY=python3
fi
exec "$PY" "$DIR/estimator.py" "$@"
