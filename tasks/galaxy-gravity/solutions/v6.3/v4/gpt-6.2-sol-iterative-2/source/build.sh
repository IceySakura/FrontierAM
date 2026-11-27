#!/bin/sh
set -eu
cd "$(dirname "$0")"
g++ -std=c++17 -O2 plan_fft.cpp -lfftw3 -o _plan_fft
./_plan_fft > fft_wisdom.hpp
rm -f _plan_fft
exec g++ -std=c++17 -O3 -march=znver2 -ffp-contract=fast -fno-math-errno \
    -fno-trapping-math -DNDEBUG solver.cpp -lfftw3 -o solve
