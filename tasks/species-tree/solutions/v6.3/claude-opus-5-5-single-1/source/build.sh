#!/bin/sh
# Offline build: two binaries (AVX2/FMA and generic x86-64); run.sh picks one.
cd "$(dirname "$0")" || exit 1
CXX=${CXX:-g++}
$CXX -O3 -std=c++17 -o stree_generic stree.cpp || exit 1
$CXX -O3 -mavx2 -mfma -std=c++17 -o stree_avx2 stree.cpp || cp stree_generic stree_avx2
exit 0
