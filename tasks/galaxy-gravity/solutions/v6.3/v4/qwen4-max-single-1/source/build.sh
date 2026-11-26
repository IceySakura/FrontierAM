#!/bin/sh
set -eu
cd "$(dirname "$0")"
g++ -O3 -march=native -funroll-loops -ffast-math -fno-math-errno -std=c++17 solve.cpp -o solve
