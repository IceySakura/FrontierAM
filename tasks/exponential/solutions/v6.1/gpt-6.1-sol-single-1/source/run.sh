#!/bin/sh
set -eu
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ ! -x ./solver ]; then
    sh ./build.sh
fi
exec ./solver
