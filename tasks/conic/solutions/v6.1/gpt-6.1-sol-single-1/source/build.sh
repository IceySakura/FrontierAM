#!/bin/sh
set -eu
export OMP_NUM_THREADS=1
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
g++ -O3 -DNDEBUG -std=c++17 -fPIC -shared "$HERE/kernels.cpp" -o "$HERE/libkernels.so.build"
mv -f "$HERE/libkernels.so.build" "$HERE/libkernels.so"
export PYTHONPYCACHEPREFIX="$HERE/.pycache"
export OPENBLAS_NUM_THREADS=1
python -c 'import argparse, ctypes, glob, importlib.util, json, sysconfig, zipfile, numpy, scipy.sparse, scipy.linalg, scipy.optimize'
python -m compileall -f -q "$HERE"
