#!/bin/sh
# Build the species-tree inference program (single translation unit, C++17).
set -e
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -o sp sp.cpp
