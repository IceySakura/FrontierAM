#!/bin/sh
# Single-threaded numpy: the task allows one logical CPU.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
# Pick an interpreter that actually has numpy (guard against PATH changes).
PY=""
for c in "$(command -v python3 2>/dev/null)" /usr/local/bin/python3 /usr/bin/python3 python3; do
    [ -n "$c" ] || continue
    if "$c" -c 'import numpy' >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -n "$PY" ] || PY=python3
exec "$PY" "$DIR/main.py" "$1" "$2"
