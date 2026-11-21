#!/bin/sh
set -eu
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
g++ -std=c++17 -O3 -ffast-math -march=native -fPIC -shared native.cpp -o native.so
