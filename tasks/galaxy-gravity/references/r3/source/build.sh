#!/bin/sh
set -eu
cd "$(dirname "$0")"
# Frozen paper-based configuration compiled in as defaults: p = 8, theta = 0.48, constant opening angle (Dehnen 2000 Eq. 2),
# no generated kernels, no kernel split (candidates/fmm-dehnen/CALIBRATION.md). fmm_unrolled.hpp is gen_unrolled.py output.
g++ -O3 -std=c++17 -march=x86-64-v3 -funroll-loops -fno-math-errno -fno-trapping-math solve.cpp -o solve
