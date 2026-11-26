#!/bin/sh
set -e
cd "$(dirname "$0")"
g++ -O3 -march=native -fno-math-errno -fno-trapping-math -funroll-loops -std=c++17 solve.cpp -o solve -L. -l:libopenblasp-r0-23e5df77.3.21.dev.so -Wl,--disable-new-dtags -Wl,-rpath,'$ORIGIN' -lm
