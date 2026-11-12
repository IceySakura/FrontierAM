#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="$HERE/pycache"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
unset PYTHONDONTWRITEBYTECODE
g++ -O3 -std=c++17 -fPIC -shared "$HERE/native_ldl.cpp" "$HERE/native_affine.cpp" -o "$HERE/native_ldl.so"
python -c 'import json, os, sys; import numpy; from scipy import sparse, linalg; import scipy.sparse.linalg; from scipy.optimize._highspy import _core; import solver, cones, ipm, polyhedral, simple, validation, reductions, fastpoly, linear, affine, polish, conic_polish, admm, numeric, boxkkt'
