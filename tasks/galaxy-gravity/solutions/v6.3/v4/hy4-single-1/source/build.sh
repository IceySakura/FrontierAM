#!/bin/sh
set -e
cd "$(dirname "$0")"
g++ -O3 -march=native -mtune=native -std=c++17 -fno-math-errno -o solve solve.cpp -lm
