#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
"${CXX:-g++}" -O3 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic \
    solver.cpp -o infer
