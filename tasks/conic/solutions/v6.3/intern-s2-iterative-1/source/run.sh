#!/bin/sh
# Conic QP solver entry point.
# Usage: ./run.sh --input INPUT_PATH --output OUTPUT_PATH [other flags]
set -eu

PY=${PY:-python3}
DIR=$(cd "$(dirname "$0")" && pwd)

exec "$PY" "$DIR/conic.py" "$@"
