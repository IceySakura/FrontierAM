#!/bin/sh
# Offline build of the species-tree inference binary.
cd "$(dirname "$0")" || exit 1
CXX=${CXX:-g++}
set -e
rm -f stree_base stree_avx2
# portable baseline binary (always built)
$CXX -O3 -std=c++17 -o stree_base stree.cpp
# optimized binary for hosts with AVX2/FMA (used only if the CPU supports it)
if grep -q avx2 /proc/cpuinfo 2>/dev/null && grep -q fma /proc/cpuinfo 2>/dev/null; then
  if $CXX -O3 -std=c++17 -mavx2 -mfma -o stree_avx2 stree.cpp 2>/dev/null; then :; else rm -f stree_avx2; fi
fi
echo "build ok" >&2
