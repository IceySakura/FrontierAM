#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"
exec "${CXX:-g++}" -O3 -DNDEBUG -std=c++17 -march=native solver.cpp -o solver
