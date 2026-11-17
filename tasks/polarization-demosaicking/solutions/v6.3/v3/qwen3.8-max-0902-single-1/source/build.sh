#!/bin/sh
# Nothing to compile: solution.py is pure NumPy.  Pre-bytecode the module so the
# per-scene process does not pay for parsing, then exit.
set -e
cd "$(dirname "$0")"
python3 -m compileall -q . >/dev/null 2>&1 || true
exit 0
