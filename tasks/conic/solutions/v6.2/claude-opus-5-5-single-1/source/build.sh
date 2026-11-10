#!/bin/bash
# Compile the native solver: one C++17 translation unit, offline, using only preinstalled
# libraries (libz.so.1, reference LAPACK/BLAS). If compilation fails, fall back to the
# prebuilt binary shipped in bin/solver.prebuilt (built from the same sources in the same image).
cd "$(dirname "$0")"
mkdir -p bin
LIBS="-l:libz.so.1 -llapack -lblas -ldl"
if g++ -O3 -std=c++17 -o bin/solver.tmp src/solver.cpp $LIBS 2>/dev/null \
   || g++ -O3 -std=c++17 -o bin/solver.tmp src/solver.cpp -l:libz.so.1 -l:liblapack.so.3 -l:libblas.so.3 -ldl; then
  mv -f bin/solver.tmp bin/solver
else
  rm -f bin/solver.tmp
  cp -f bin/solver.prebuilt bin/solver
fi
chmod +x bin/solver
exit 0
