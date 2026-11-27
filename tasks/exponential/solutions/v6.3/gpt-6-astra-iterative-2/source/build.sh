#!/bin/sh
set -eu
cd "$(dirname "$0")"
g++ -std=c++17 -O3 -march=x86-64-v3 -ffp-contract=fast -DNDEBUG \
    solver.cpp -o solver -llapacke -lopenblas -l:libumfpack.so.5 -l:libfftw3.so.3
