#!/bin/sh
# Entry point: ./run.sh --input INPUT.npz --output OUTPUT.npz
DIR="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec python3 "$DIR/solver.py" "$@"
