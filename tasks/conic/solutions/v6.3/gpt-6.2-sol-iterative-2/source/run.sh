#!/bin/sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
dir=${0%/*}
if [ "$dir" = "$0" ]; then dir=.; fi
if [ -d "$dir/.pylibcache" ]; then
    export PYTHONPYCACHEPREFIX="$dir/.pylibcache"
fi
if [ -x "$dir/conic-fast" ]; then
    exec "$dir/conic-fast" "$dir/main.py" "$@"
fi
exec python "$dir/main.py" "$@"
