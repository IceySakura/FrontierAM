#!/bin/sh
set -eu
cd "$(dirname "$0")"
g++ -O3 -std=c++17 -march=native -ffp-contract=fast -fno-math-errno solver.cpp -o solve
