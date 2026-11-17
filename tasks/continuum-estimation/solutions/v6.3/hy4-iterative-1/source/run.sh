#!/bin/sh
set -eu
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS PYTHONDONTWRITEBYTECODE
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -x /workspace/runtime/bin/python3 ]; then
    PY=/workspace/runtime/bin/python3
else
    PY=python3
fi
exec "$PY" "$HERE/main.py" "$@"
