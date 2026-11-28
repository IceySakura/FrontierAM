#!/bin/sh
# Continuum limit estimation.  No arguments: read /workspace/inputs/input.json
# and write /workspace/result/output.json.  Optional: run.sh INPUT OUTPUT.
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ "$#" -gt 2 ]; then
  echo "usage: run.sh [INPUT [OUTPUT]]" >&2
  exit 2
fi
IN=${1:-/workspace/inputs/input.json}
OUT=${2:-/workspace/result/output.json}
PY=/workspace/runtime/bin/python3
if [ ! -x "$PY" ]; then PY=python3; fi
command -v python3 >/dev/null 2>&1 || PY=/workspace/runtime/bin/python3
exec "$PY" "$HERE/estimate.py" "$IN" "$OUT"
