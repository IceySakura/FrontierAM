#!/bin/sh
# Usage: ./run.sh INPUT OUTPUT
#
# Reads a Newick file of estimated gene trees (one per line) and writes a
# single unrooted binary species tree to OUTPUT.  Only the Python standard
# library is required.
set -e
# The task allows a single logical CPU; make sure no library starts threads.
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS
dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$dir/stree.py" "$1" "$2"
