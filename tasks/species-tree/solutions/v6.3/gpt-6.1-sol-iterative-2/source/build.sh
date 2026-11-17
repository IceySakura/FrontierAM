#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec g++ -O3 -std=c++17 -DNDEBUG main.cpp -o infer
