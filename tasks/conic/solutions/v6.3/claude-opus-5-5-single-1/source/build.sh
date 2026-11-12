#!/bin/sh
# Precompile Python bytecode so per-instance runs skip compilation.
cd "$(dirname "$0")" && python3 -m compileall -q -f --invalidation-mode unchecked-hash solver.py main.py
exit 0
