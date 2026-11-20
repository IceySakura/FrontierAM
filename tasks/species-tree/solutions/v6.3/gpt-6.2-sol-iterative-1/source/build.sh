#!/bin/sh
set -eu
cd "$(dirname "$0")"
${CXX:-g++} -std=c++17 -O3 -march=native -DNDEBUG infer.cpp -o infer
