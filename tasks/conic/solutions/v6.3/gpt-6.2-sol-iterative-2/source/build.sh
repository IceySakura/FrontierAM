#!/bin/sh
set -eu
cd "$(dirname "$0")"
CXX=${CXX:-g++}
"$CXX" -std=c++17 -O3 -march=native -DNDEBUG -fPIC -shared native.cpp flow.cpp diagqp.cpp band.cpp components.cpp -o libconic.so
"$CXX" -std=c++17 -O3 -march=native -DNDEBUG front.cpp -Wl,-l:libarchive.so.13 -Wl,-l:libjansson.so.4 -o conic-fast
unset PYTHONDONTWRITEBYTECODE
export PYTHONPYCACHEPREFIX="$PWD/.pylibcache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
python -c 'import scipy.optimize._highspy._core, scipy.sparse.linalg; import fastlp, main, ipm, presolve, polish, poly, split, dense, chordal, pencil, cuts, alm, dualize, components, nullspace, semismooth, qpvertex, qpdual; import zipfile'
