#!/bin/sh
# Continuum-estimation entry point.
#   sh run.sh              -> reads /workspace/inputs/input.json,
#                             writes /workspace/result/output.json
#   sh run.sh INPUT OUTPUT -> same, explicit paths (local development)
set -u
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
IN=${1:-/workspace/inputs/input.json}
OUT=${2:-/workspace/result/output.json}

for PY in ${CONTINUUM_PYTHON:-} /workspace/runtime/bin/python3 python3 python3.11 /usr/local/bin/python3; do
  [ -n "$PY" ] || continue
  command -v "$PY" >/dev/null 2>&1 || continue
  "$PY" "$HERE/solver.py" "$IN" "$OUT" 2>/dev/null && exit 0
done
exit 0
