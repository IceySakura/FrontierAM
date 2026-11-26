#!/bin/sh
# Continuum estimation. No-arg paths follow the task contract; optional
# INPUT OUTPUT arguments are accepted for local development.
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
INPUT="${1:-/workspace/inputs/input.json}"
OUTPUT="${2:-/workspace/result/output.json}"
# Single thread, offline bundled numerical runtime (python3 first on PATH).
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
exec python3 "$HERE/estimate.py" "$INPUT" "$OUTPUT"
