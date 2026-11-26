#!/bin/sh
# Continuum limit estimation - entry point.
#
#   run.sh [INPUT [OUTPUT]]
#
# Without arguments the program reads /workspace/inputs/input.json and writes
# /workspace/result/output.json, as required by the evaluation harness.
set -eu

DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
IN=${1:-/workspace/inputs/input.json}
OUT=${2:-/workspace/result/output.json}

# single threaded, deterministic numerics
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS \
       NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS

# Pick an interpreter that can import the bundled numerical stack.  The
# evaluation harness puts the bundled interpreter first on PATH, the explicit
# paths are only fallbacks for local development.
PY=""
if [ -n "${CONTINUUM_PYTHON:-}" ]; then
    CANDIDATES="${CONTINUUM_PYTHON} /workspace/runtime/bin/python3 python3 /usr/local/bin/python3"
else
    CANDIDATES="/workspace/runtime/bin/python3 python3 /usr/local/bin/python3"
fi
for CAND in $CANDIDATES; do
    command -v "$CAND" >/dev/null 2>&1 || continue
    # cheap probe: is numpy installed for this interpreter?  (no import, so
    # the check costs only the interpreter start-up)
    if "$CAND" -c "import os,sys; sys.exit(0 if any(os.path.isdir(os.path.join(p,'numpy')) for p in sys.path) else 1)" \
        >/dev/null 2>&1; then
        PY="$CAND"
        break
    fi
done
if [ -z "$PY" ]; then
    PY=python3
fi

mkdir -p -- "$(dirname -- "$OUT")"

exec "$PY" "$DIR/continuum.py" "$IN" "$OUT"
