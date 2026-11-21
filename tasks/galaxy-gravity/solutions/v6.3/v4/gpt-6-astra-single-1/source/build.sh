#!/bin/sh
set -eu
cd "$(dirname "$0")"
python3 gen.py
g++ -O3 -std=c++17 -mavx2 -mfma -fno-math-errno -fno-trapping-math -DNDEBUG main.cpp -o solve
