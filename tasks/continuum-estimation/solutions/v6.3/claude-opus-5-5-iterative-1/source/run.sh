#!/bin/sh
# Usage: run.sh [INPUT [OUTPUT]]  (defaults: /workspace/inputs/input.json -> /workspace/result/output.json)
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PY=python3
if [ -x /workspace/runtime/bin/python3 ]; then PY=/workspace/runtime/bin/python3; fi
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$PY" "$HERE/solve.py" "$@"
