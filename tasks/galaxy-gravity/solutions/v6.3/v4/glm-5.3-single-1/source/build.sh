#!/bin/sh
set -e
cd "$(dirname "$0")"
CC=${CC:-cc}
OPT="-O3 -fno-math-errno -fopenmp-simd 2>/dev/null || true"
if $CC -O3 -mavx2 -mfma -fno-math-errno -o solve solve.c -lm 2>/dev/null; then
  exit 0
fi
$CC -O3 -msse2 -fno-math-errno -o solve solve.c -lm
