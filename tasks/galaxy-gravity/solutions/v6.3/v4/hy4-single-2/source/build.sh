#!/bin/sh
set -e
cd "$(dirname "$0")"
g++ -O3 -std=c++17 -march=x86-64-v3 -fno-math-errno -funroll-loops \
    -o solve solve.cpp -lm
