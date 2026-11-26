#!/bin/sh
# Entry point: ./run.sh --input INPUT.npz --output OUTPUT.npz
DIR=$(cd "$(dirname "$0")" && pwd)
exec python3 "$DIR/solver.py" "$@"
