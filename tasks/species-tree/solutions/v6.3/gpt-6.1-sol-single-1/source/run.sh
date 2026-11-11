#!/bin/sh
set -eu
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
exec ./solver "$@"
