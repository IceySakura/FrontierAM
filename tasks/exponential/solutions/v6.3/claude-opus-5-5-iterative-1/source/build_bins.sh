#!/bin/bash
# Rebuilds the two shipped binaries from expo.cpp (used during development).
set -e
cd "$(dirname "$0")"
g++ -O3 -march=haswell -mtune=znver2 -std=c++17 -o expo_avx2 expo.cpp -llapack
g++ -O3 -std=c++17 -o expo_generic expo.cpp -llapack
