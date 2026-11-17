#!/bin/sh
# Continuum-limit estimator entry point.
# No arguments: reads /workspace/inputs/input.json and writes /workspace/result/output.json.
# Optional development arguments: INPUT OUTPUT.
set -eu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -x /workspace/runtime/bin/python3 ]; then
    PY=/workspace/runtime/bin/python3
else
    PY=python3
fi
exec "$PY" "$HERE/solve.py" "$@"
