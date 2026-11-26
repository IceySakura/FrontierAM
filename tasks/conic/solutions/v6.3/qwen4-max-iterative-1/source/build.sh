#!/bin/sh
set -eu
cd "$(dirname "$0")"
g++ -O3 -DNDEBUG -std=c++17 -march=native -mtune=native \
    src/solver.cpp -o solver \
    -Llib -l:libscipy_openblas-b75cc656.so -l:libz.so.1 \
    -Wl,-rpath,'$ORIGIN/lib' -lm
python - <<'PY'
import numpy
import sys
sys.path.insert(0, "/usr/local/lib/python3.12/site-packages/scipy/optimize")
from _highspy import _core
import scipy.sparse.linalg
PY
