#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"
g++ -std=c++17 -O3 -march=native -ffast-math -fno-finite-math-only \
    -shared -fPIC native.cpp -lfftw3f -o deconv_native.so
