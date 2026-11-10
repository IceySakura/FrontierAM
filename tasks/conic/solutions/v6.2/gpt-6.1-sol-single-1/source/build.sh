#!/bin/sh
set -eu
cd "$(dirname "$0")"
g++ -O3 -DNDEBUG -std=c++17 -fPIC -shared native.cpp -o native.so
unset PYTHONDONTWRITEBYTECODE
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="$PWD/pycache"
python -c "import numpy, scipy.sparse.linalg, scipy.linalg, scipy.optimize._highspy._core; import solver, cones, ipm, ldl, presolve, barrier, gp, lasso, soc, edge, admm, qp_lp, soc_lp"
