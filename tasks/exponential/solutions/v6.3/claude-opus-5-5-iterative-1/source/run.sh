#!/bin/bash
# FAMEXP2 entry point: picks the AVX2 build when the CPU supports it and it loads correctly.
cd "$(dirname "$0")" || exit 1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 GOTO_NUM_THREADS=1 MKL_NUM_THREADS=1
if grep -qw avx2 /proc/cpuinfo 2>/dev/null && grep -qw fma /proc/cpuinfo 2>/dev/null \
   && [ -x ./expo_avx2 ] && ./expo_avx2 --selftest </dev/null >/dev/null 2>&1; then
  exec ./expo_avx2
fi
if [ -x ./expo_generic ] && ./expo_generic --selftest </dev/null >/dev/null 2>&1; then
  exec ./expo_generic
fi
# last resort: build from source (before READY, so not timed)
g++ -O2 -std=c++17 -o /tmp/expo_built expo.cpp -llapack 1>&2 && exec /tmp/expo_built
exit 1
