#!/bin/sh
set -eu
cd "$(dirname "$0")"
g++ -O3 -std=c++17 -march=znver2 -ffast-math -fno-finite-math-only -fno-tree-slp-vectorize -o solve solver.cpp
