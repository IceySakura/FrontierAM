#!/bin/bash
set -e
cd "$(dirname "$0")"
gcc -O3 -march=native -mtune=native -ffast-math -fno-math-errno -Wall -o solve solve.c -lm
