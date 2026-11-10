#!/bin/sh
set -eu
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
g++ -O3 -DNDEBUG -std=c++17 -fPIC -shared "$HERE/native_ldl.cpp" -o "$HERE/libldl.so"
python -X "pycache_prefix=$HERE/pycache" "$HERE/prepare.py"
