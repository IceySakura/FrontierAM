#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
g++ -O3 -DNDEBUG -std=c++17 -march=x86-64-v3 -ffp-contract=fast \
    -fassociative-math -fno-signed-zeros -fno-trapping-math \
    solver.cpp -o solver
