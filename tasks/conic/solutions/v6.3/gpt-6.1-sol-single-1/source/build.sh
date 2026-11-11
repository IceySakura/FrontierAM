#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="$HERE/.pycache"
unset PYTHONDONTWRITEBYTECODE
python3 -c 'import os, sys; sys.path.insert(0, os.path.dirname(os.environ["PYTHONPYCACHEPREFIX"])); import solver, cones, accuracy, polyhedral, interior, refine, presolve'
