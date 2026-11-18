#!/bin/sh
# Continuum estimation entry point.
#   /bin/sh run.sh              -> reads /workspace/inputs/input.json,
#                                  writes /workspace/result/output.json
#   /bin/sh run.sh INPUT OUTPUT -> local development form
set -u

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" 2>/dev/null && pwd)
[ -n "$HERE" ] || HERE=/workspace/submission

if [ -x /workspace/runtime/bin/python3 ]; then
  PY=/workspace/runtime/bin/python3
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=/usr/local/bin/python3
fi

if [ "$#" -ge 2 ]; then
  exec "$PY" "$HERE/solve.py" "$1" "$2"
fi
exec "$PY" "$HERE/solve.py"
