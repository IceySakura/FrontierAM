#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec g++ -std=c++17 -O3 -march=x86-64-v3 -fno-math-errno -fPIC -shared \
    -Wall -Wextra -DNDEBUG solver.cpp -lfftw3f -o solver.so
