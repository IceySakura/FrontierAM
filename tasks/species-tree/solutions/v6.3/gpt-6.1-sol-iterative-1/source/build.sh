#!/bin/sh
set -eu
cd "$(dirname "$0")"
g++ -O3 -DNDEBUG -std=c++17 main.cpp -o infer
