#!/bin/sh
# FAMEXP2 entry point.  Prefers the natively built solver, falls back to the
# pure numpy/scipy implementation if no binary is present.
cd "$(dirname "$0")" || exit 1
if [ -x bin/famexp2 ]; then
    exec ./bin/famexp2
fi
if command -v python3 >/dev/null 2>&1; then
    exec python3 ./solver_py.py
fi
echo "no solver available" >&2
exit 1
