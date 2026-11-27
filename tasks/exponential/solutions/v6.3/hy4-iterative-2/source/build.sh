#!/bin/bash
set -e
cd "$(dirname "$0")"
g++ -O3 -std=c++17 -funroll-loops -o solver solver.cpp -lm
