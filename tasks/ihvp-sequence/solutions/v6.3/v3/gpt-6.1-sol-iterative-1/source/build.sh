#!/bin/sh
set -eu
cd "$(dirname "$0")"
"${CC:-gcc}" -O3 -march=native -fopenmp-simd -fPIC -shared rotations.c -o _rotations.so
