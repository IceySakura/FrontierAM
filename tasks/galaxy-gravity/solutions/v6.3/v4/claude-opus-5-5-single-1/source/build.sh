#!/bin/sh
set -e
cd "$(dirname "$0")"
g++ -O3 -mavx2 -mfma -mtune=znver2 -fno-math-errno -fno-trapping-math -fassociative-math -fno-signed-zeros -freciprocal-math -std=c++17 -o solve solve.cpp
