#!/bin/sh
# Entry point: ./run.sh --input IN.npz --output OUT.npz
DIR=$(cd "$(dirname "$0")" && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "$DIR/bin/solver" "$@"
