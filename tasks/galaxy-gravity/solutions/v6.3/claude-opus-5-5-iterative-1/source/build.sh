#!/bin/sh
# Build ./solve (AVX2+FMA, single-threaded). Generated derivative code (gen_derivs*.inc) is shipped pre-generated.
# If compilation fails for any reason, a previously built ./solve (if present) is kept.
cd "$(dirname "$0")"
if g++ -O2 -mavx2 -mfma -std=c++17 -o solve.tmp solve.cpp; then
    mv -f solve.tmp solve
fi
[ -x solve ]
