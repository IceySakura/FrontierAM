#!/bin/sh
# Single threaded numerics (required by the task) and no BLAS thread pool.
here="$(cd "$(dirname "$0")" && pwd)"
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
PYTHONHASHSEED=0
export OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS PYTHONHASHSEED
PY=python3
command -v python3 >/dev/null 2>&1 || PY=python
exec "$PY" -B "$here/solver.py" "$@"
