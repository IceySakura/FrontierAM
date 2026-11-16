#!/bin/sh
set -e
cd "$(dirname "$0")"
g++ -O3 -mavx2 -mfma -mtune=znver2 -std=c++17 -o solve solve.cpp
