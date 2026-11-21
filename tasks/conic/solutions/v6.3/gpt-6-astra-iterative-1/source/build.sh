#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPYCACHEPREFIX="$ROOT/.pycache"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset PYTHONDONTWRITEBYTECODE
cd "$ROOT"
gcc -O3 -std=c99 -fPIC -shared ldl.c -o libconic_ldl.so -lm
exec python -c 'import entry, bootstrap, fast_lp, solver, ipm, qp_pd, highs_solver, presolve, poly_solver, poly_kkt, box_kkt, box_solver, refine, dualize, structured, relaxation, native_ldl; import scipy.optimize._highspy._core; import zipfile; import numpy.linalg'
