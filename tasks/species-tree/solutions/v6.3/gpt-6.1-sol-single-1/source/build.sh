#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec g++ -std=c++17 -O3 -DNDEBUG -march=x86-64 -mtune=generic main.cpp -o solver
