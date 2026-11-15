#!/bin/sh
# Offline build. No network, no dependencies beyond a C++ compiler.
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
CXX=${CXX:-g++}
"$CXX" -O3 -std=c++11 -static-libstdc++ -o "$DIR/species" "$DIR/species.cpp" 2>/dev/null \
  || "$CXX" -O3 -std=c++11 -o "$DIR/species" "$DIR/species.cpp"
