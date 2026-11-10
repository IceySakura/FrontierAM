#!/bin/sh
set -e
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -o solver src/solver.cpp /usr/lib/x86_64-linux-gnu/libz.so.1 -llapack -lblas
