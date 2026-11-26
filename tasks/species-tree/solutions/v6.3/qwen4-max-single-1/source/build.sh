#!/bin/sh
set -e
cd "$(dirname "$0")"
CXX="${CXX:-g++}"
"$CXX" -O3 -mpopcnt -std=c++17 -o species src/species.cpp
