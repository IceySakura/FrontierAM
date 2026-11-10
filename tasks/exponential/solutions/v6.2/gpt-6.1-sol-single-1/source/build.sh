#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
g++ -O3 -std=c++17 -march=x86-64-v3 -fno-math-errno -DNDEBUG solver.cpp -o solver
