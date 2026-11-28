#!/bin/bash
set -euo pipefail
g++ -O3 -std=c++17 -march=x86-64 -mtune=native -o solver solver.cpp -llapacke -llapack -lopenblas -lm
