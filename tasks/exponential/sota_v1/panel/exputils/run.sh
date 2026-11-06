#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

export HOME=/tmp
export JULIA_DEPOT_PATH="$HERE/depot"
export JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

exec "$HERE/julia-1.11.7/bin/julia" \
    --startup-file=no --history-file=no --project="$HERE" \
    --sysimage="$HERE/exputils_sysimage_min.so" \
    "$HERE/stream_exputils.jl" "${1:-0.000001}"
