#!/bin/sh
set -eu
cd "$(dirname "$0")"
g++ -O3 -std=c++17 -mpopcnt -o species_tree species_tree.cpp
