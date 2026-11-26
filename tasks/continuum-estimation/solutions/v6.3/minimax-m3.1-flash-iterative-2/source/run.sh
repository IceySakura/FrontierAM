#!/bin/sh
# Continuum estimation from positive discretization levels.
#
# Usage:  run.sh [INPUT_JSON] [OUTPUT_JSON]
# Defaults: /workspace/inputs/input.json -> /workspace/result/output.json
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
INPUT=${1:-/workspace/inputs/input.json}
OUTPUT=${2:-/workspace/result/output.json}

# Pick an interpreter that can import NumPy: the verifier puts the bundled
# runtime first on PATH, but the explicit runtime path is used as a fallback.
PY=""
for CAND in "${CONTINUUM_PYTHON:-}" python3 /workspace/runtime/bin/python3 \
            /opt/continuum-runtime/bin/python3 /usr/local/bin/python3; do
    [ -n "$CAND" ] || continue
    command -v "$CAND" >/dev/null 2>&1 || continue
    if "$CAND" -c "import numpy" >/dev/null 2>&1; then
        PY="$CAND"
        break
    fi
done
if [ -z "$PY" ]; then
    # no NumPy available: continuum.py falls back to a pure-python estimator
    PY=python3
    command -v "$PY" >/dev/null 2>&1 || PY=/workspace/runtime/bin/python3
fi

exec "$PY" "$HERE/continuum.py" "$INPUT" "$OUTPUT"
