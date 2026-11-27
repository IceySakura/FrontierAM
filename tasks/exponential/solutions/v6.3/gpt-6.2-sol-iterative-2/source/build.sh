#!/bin/sh
set -eu
cd "$(dirname "$0")"
${CXX:-g++} -O3 -std=c++20 -march=x86-64-v3 -mtune=znver2 \
    -ffp-contract=off -fno-math-errno -Wall -Wextra \
    main.cpp \
    -Wl,-l:libfftw3.so.3 -lmvec -o propagate
