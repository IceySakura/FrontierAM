#!/bin/sh
set -eu
cd "$(dirname "$0")"
g++ -std=c++17 -O3 -march=x86-64-v3 -ffp-contract=fast \
    -DNDEBUG -I/usr/include/eigen3 solver.cpp -o solver
