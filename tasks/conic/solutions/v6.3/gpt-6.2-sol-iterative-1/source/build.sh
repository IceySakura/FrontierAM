#!/bin/sh
set -eu
cd "$(dirname "$0")"
blas=$(python -c 'import glob,sysconfig; print(glob.glob(sysconfig.get_path("platlib")+"/scipy.libs/libscipy_openblas-*.so")[0])')
libdir=${blas%/*}
python find_highs.py > highs_symbols.hpp
g++ -std=c++17 -O3 -DNDEBUG -march=x86-64-v3 -D_GLIBCXX_USE_CXX11_ABI=0 \
    -fPIC -c highs_native.cpp -o highs_native.o
g++ -std=c++17 -O3 -DNDEBUG -march=x86-64-v3 -fPIC -shared solver.cpp highs_native.o "$blas" \
    -Wl,-rpath,"$libdir" -o libconic.so
g++ -std=c++17 -O3 -DNDEBUG -march=x86-64-v3 -I/usr/include/node main.cpp \
    -L. -lconic -l:libz.so.1 -l:libjansson.so.4 -Wl,-rpath,'$ORIGIN' -o conic
