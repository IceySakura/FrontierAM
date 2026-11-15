#!/bin/sh
# Offline build: compiles the single-file C++17 implementation. No network needed.
set -e
cd "$(dirname "$0")"
CXX=${CXX:-g++}
"$CXX" -O2 -std=c++17 -o species_tree species_tree.cpp
