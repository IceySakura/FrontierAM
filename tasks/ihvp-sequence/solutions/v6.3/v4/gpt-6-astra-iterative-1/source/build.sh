#!/bin/sh
set -eu
cd "$(dirname "$0")"
cc -O3 -march=native -fopenmp-simd -fPIC -shared kernels.c -o _kernels.so -lm
