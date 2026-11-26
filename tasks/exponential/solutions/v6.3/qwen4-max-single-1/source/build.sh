#!/bin/sh
set -eu
cd "$(dirname "$0")"
g++ -O3 -march=native -funroll-loops -shared -fPIC fame_core.cpp -o fame_core.so
