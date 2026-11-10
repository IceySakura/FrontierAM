#!/bin/sh
set -eu
cd "$(dirname "$0")"
python3 generate.py 10 kernels.hpp 10
python3 harmonic_gen.py 16 harmonic.hpp
exec g++ -std=c++17 -O2 -fno-tree-slp-vectorize -march=native -ffast-math -fno-math-errno -o solve solver.cpp
