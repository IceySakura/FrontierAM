#!/bin/sh
set -eu
cd "$(dirname "$0")"
g++ -O3 -std=c++17 -march=native -ffast-math -fno-finite-math-only -DUSE_EXACT_FORCE -o solve hybrid.cpp
