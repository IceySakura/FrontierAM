#!/bin/sh
set -eu
base=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
g++ -O3 -std=c++17 -DNDEBUG "$base/native.cpp" -l:libz.so.1 -o "$base/native"
exec g++ -O3 -std=c++17 -DNDEBUG -fPIC -shared "$base/sparse_ldl.cpp" -o "$base/sparse_ldl.so"
