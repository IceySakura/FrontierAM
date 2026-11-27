#!/bin/bash
cd "$(dirname "$0")"
if [ ! -x ./solver ]; then
    g++ -O3 -std=c++17 -funroll-loops -o solver solver.cpp -lm
fi
exec ./solver
