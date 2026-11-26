#!/bin/sh
# FAMEXP2 submission entry point: reads the session on stdin, writes answers on stdout.
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
for PY in python3 /usr/local/bin/python3 python; do
    if command -v "$PY" >/dev/null 2>&1; then
        exec "$PY" "$DIR/main.py"
    fi
done
echo "no python interpreter found" >&2
exit 1
