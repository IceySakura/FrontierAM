#!/bin/sh
# Build the solver; fall back to the prebuilt binary if compilation fails.
cd "$(dirname "$0")"
if g++ -O3 -mavx2 -mfma -mtune=znver2 -o solve.tmp fmm.cpp 2>build.log; then
  mv -f solve.tmp solve
else
  cp -f solve.prebuilt solve
fi
chmod +x solve
