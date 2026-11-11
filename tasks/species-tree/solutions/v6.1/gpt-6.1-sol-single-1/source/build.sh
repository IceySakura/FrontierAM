#!/bin/sh
set -eu
cd "$(dirname "$0")"
g++ -O3 -DNDEBUG -std=c++17 infer.cpp -o infer
