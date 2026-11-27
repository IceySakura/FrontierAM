#!/bin/sh
set -eu
cd "$(dirname "$0")"
${CXX:-c++} -std=c++17 -O3 -march=x86-64-v3 -DNDEBUG -Wall -Wextra -o infer main.cpp
