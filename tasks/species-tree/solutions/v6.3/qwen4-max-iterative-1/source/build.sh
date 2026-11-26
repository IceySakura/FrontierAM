#!/bin/sh
set -eu
cd "$(dirname "$0")"
c++ -O3 -DNDEBUG -std=c++17 -o species_tree main.cpp
