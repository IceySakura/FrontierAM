#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="$HERE/.cache"
unset PYTHONDONTWRITEBYTECODE
cd "$HERE"
python -c 'import argparse, json, traceback, zipfile; import numpy; import scipy.sparse.linalg; import scipy.linalg; import scipy.optimize._highspy._core; import common, cones, highs_solver, ipm, reduce, separable, polish, main'
