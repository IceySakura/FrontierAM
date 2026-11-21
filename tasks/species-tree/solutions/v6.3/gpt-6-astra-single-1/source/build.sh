#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec g++ -O3 -DNDEBUG -std=c++17 -march=x86-64 -mtune=generic solver.cpp -o solver
