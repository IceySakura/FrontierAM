#!/bin/sh
set -eu
cd "$(dirname -- "$0")"
"${CXX:-g++}" -O3 -DNDEBUG -std=c++17 -fPIC -shared innovations.cpp -o libinnovations.so
