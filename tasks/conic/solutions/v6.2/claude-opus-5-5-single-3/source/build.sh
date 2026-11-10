#!/bin/sh
set -e
cd "$(dirname "$0")"
mkdir -p bin
g++ -O3 -std=c++17 -fno-math-errno -o bin/conic_solver src/solver.cpp -l:libz.so.1
