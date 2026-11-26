#!/bin/sh
# Entry point for the continuum-limit estimator.
#
#   run.sh [INPUT_JSON [OUTPUT_JSON]]
#
# With no arguments it reads /workspace/inputs/input.json and writes
# /workspace/result/output.json, as required by the task specification.
# The bundled numerical runtime (Python 3.11 + NumPy) is used when present;
# otherwise the interpreter found on PATH is used.

set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
IN=${1:-/workspace/inputs/input.json}
OUT=${2:-/workspace/result/output.json}

PY=""
for candidate in /workspace/runtime/bin/python3 /opt/continuum-runtime/bin/python3; do
    if [ -x "$candidate" ]; then
        PY="$candidate"
        break
    fi
done
if [ -z "$PY" ]; then
    PY=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)
fi
if [ -z "$PY" ]; then
    echo "no python interpreter available" >&2
    exit 1
fi

mkdir -p "$(dirname -- "$OUT")"
exec "$PY" "$HERE/estimate.py" "$IN" "$OUT"
