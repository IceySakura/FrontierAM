#!/bin/sh
set -e
cd "$(dirname "$0")"
gcc -O3 -mavx2 -mfma -funroll-loops -fno-math-errno -o solve solve.c -lm
chmod +x solve
