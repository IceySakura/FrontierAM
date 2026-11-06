#!/bin/sh
set -eu
cd "$(dirname "$0")"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [ "$#" -gt 1 ]; then
    echo "usage: $0 [tolerance-multiplier]" >&2
    exit 2
fi
exec ./stream_slepc "${1:-0.001}"
