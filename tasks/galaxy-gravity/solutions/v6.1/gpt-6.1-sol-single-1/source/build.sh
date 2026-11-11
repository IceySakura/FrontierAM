#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"
exec g++ -O3 -march=native -ffast-math -std=c++17 -DNDEBUG main.cpp -o solve
