#!/bin/sh
# Entry point: ./run.sh --input INPUT_PATH --output OUTPUT_PATH
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$DIR/solver.py" "$@"
