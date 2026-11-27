#!/bin/bash
set -eu
cd "$(dirname "$0")"
g++ -std=c++17 -O3 -march=x86-64-v3 -fno-math-errno -ffp-contract=fast -DFAMEXP_FORCE_INLINE \
    -I/usr/include/eigen3 solver.cpp -o solver -llapacke -lopenblas -l:libfftw3.so.3
