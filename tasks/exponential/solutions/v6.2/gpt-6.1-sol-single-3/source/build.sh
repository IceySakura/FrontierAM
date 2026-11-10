#!/bin/sh
set -eu
cd "$(dirname "$0")"
g++ -O3 -DNDEBUG -std=c++17 -DEIGEN_DONT_PARALLELIZE -I/usr/include/eigen3 solver.cpp -o solver
