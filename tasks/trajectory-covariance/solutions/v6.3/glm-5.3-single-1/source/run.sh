#!/bin/sh
# run.sh INPUT OUTPUT   ->   writes the d x d long-run covariance estimate
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 SEED=1
exec python3 "$DIR/estimator.py" "$1" "$2"
