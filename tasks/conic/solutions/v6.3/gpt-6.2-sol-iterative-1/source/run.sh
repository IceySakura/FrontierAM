#!/bin/sh
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 BLIS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
ulimit -v 3543040 2>/dev/null || :
if [ "${CONIC_NATIVE:-1}" = 1 ]; then
    exec "$(dirname "$0")/conic" "$@"
fi
exec python "$(dirname "$0")/run.py" "$@"
