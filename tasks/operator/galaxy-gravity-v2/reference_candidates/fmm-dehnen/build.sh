#!/bin/sh
set -eu
cd "$(dirname "$0")"
g++ -O3 -std=c++17 -march=x86-64-v3 -funroll-loops -fno-math-errno -fno-trapping-math solve.cpp -o solve
