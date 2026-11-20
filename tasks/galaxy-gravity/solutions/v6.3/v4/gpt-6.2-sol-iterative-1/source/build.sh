#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec g++ -std=c++17 -O3 -march=znver2 -ffp-contract=fast -fno-math-errno -fno-trapping-math -DNDEBUG solver.cpp -o solve
