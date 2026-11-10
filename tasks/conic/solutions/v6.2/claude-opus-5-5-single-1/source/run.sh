#!/bin/bash
# Usage: ./run.sh --input INPUT.npz --output OUTPUT.npz
DIR="$(cd "$(dirname "$0")" && pwd)"
export OPENBLAS_NUM_THREADS=1
if [ -x "$DIR/bin/solver" ]; then
  exec "$DIR/bin/solver" "$@"
fi
exec "$DIR/bin/solver.prebuilt" "$@"
