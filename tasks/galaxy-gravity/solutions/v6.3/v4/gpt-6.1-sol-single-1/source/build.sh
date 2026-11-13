#!/bin/sh
set -eu
cd "$(dirname "$0")"
python3 generate_kernels.py
exec g++ -std=c++17 -O3 -march=x86-64-v3 -ffast-math -fno-math-errno -DNDEBUG solver.cpp -o solve
