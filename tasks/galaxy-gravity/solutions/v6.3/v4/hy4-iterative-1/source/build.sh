#!/bin/sh
set -e
cd "$(dirname "$0")"
g++ -O3 -mavx2 -mfma -ffp-contract=fast -fno-math-errno -std=c++17 \
    -o solve solve.cpp
chmod +x solve
