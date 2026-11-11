#!/bin/sh
set -eu
cd "$(dirname "$0")"
c++ -O3 -DNDEBUG -std=c++17 -march=native -ffp-contract=fast -I/usr/include/eigen3 solver.cpp -o solver
