#!/bin/sh
# Usage: sh run.sh [INPUT [OUTPUT]]  (defaults: /workspace/inputs/input.json, /workspace/result/output.json)
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
IN=${1:-/workspace/inputs/input.json}
OUT=${2:-/workspace/result/output.json}
PY=python3
if [ -x /workspace/runtime/bin/python3 ]; then PY=/workspace/runtime/bin/python3; fi
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$(dirname -- "$OUT")"
exec "$PY" "$HERE/continuum.py" "$IN" "$OUT"
