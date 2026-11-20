#!/bin/bash
set -eu
cd "$(dirname "$0")"
out=./propagate.build
trap 'rm -f "$out"' EXIT
g++ -std=c++17 -O3 -march=x86-64-v3 -mtune=znver2 \
    -fno-math-errno -Wall -Wextra -Wpedantic -DFAM_SMALL_BLOCKS \
    -fopenmp-simd -DFAM_ENABLE_DEFLATE -DFAM_TEMPORAL_BANDS \
    -DFAM_ENABLE_STEADY -DFAM_RECENTER solver.cpp -lmvec -o "$out"
mv -f "$out" ./propagate
