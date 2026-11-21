#!/bin/sh
set -eu
cd "$(dirname "$0")"
g++ -O3 -std=c++17 -march=x86-64-v3 -ffp-contract=fast -DNDEBUG solver.cpp -o solver
