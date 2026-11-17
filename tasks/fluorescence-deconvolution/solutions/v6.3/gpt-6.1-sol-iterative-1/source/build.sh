#!/bin/sh
set -eu
cd "$(dirname "$0")"
if command -v g++ >/dev/null 2>&1; then
    g++ -O3 -std=c++17 -fPIC -shared -fno-math-errno -ffp-contract=fast -fopenmp-simd solver.cpp -o solver.so
    g++ -O3 -std=c++17 -fPIC -shared -fno-math-errno -ffp-contract=fast -fopenmp-simd -mavx2 -mfma solver.cpp -o solver_fast.so
else
    test -s solver.so
    test -s solver_fast.so
fi
