#!/bin/sh
set -eu
cd "$(dirname "$0")"
CXX="${CXX:-g++}"
INCLUDE="$(python3 -c 'import numpy; print(numpy.get_include())')"
SUFFIX="$(python3-config --extension-suffix)"
"$CXX" -O3 -march=native -std=c++17 -fPIC -fopenmp-simd -shared \
    $(python3-config --includes) -I"$INCLUDE" core.cpp -o "spd_core$SUFFIX"
