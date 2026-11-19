#!/bin/sh
set -eu

# Directory this script lives in, resolved against the script itself rather
# than against the (unspecified) process cwd.
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)

# Input/output paths, in order of precedence:
#   1. explicit positional arguments
#   2. INPUT / OUTPUT environment variables
#   3. the task-specified defaults
IN=${1:-${INPUT:-/workspace/inputs/input.json}}
OUT=${2:-${OUTPUT:-/workspace/result/output.json}}

# /workspace/runtime/bin is first on PATH for the verifier, so the bare
# python3 below resolves to the bundled interpreter (numpy/scipy inside).
exec python3 "$SCRIPT_DIR/estimate.py" "$IN" "$OUT"
