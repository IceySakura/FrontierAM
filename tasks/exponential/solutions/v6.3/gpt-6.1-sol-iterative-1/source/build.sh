#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
g++ -O3 -std=c++17 -march=x86-64-v3 -ffp-contract=fast -DNDEBUG -I/usr/include/eigen3 \
    solver.cpp -o solver -lopenblas
