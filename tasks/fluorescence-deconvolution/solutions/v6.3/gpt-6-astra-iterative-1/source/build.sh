#!/bin/sh
set -eu
cd "$(dirname "$0")"
g++ -O3 -ffast-math -march=native -fPIC -shared kernels.cpp -o kernels.so
