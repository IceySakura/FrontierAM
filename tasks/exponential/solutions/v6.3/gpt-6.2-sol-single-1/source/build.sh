#!/bin/sh
set -eu
cd "$(dirname "$0")"
target=famexp.tmp.$$
trap 'rm -f "$target"' EXIT HUP INT TERM
c++ -std=c++17 -O3 -DNDEBUG -march=x86-64-v3 -mtune=znver2 \
    -ffp-contract=fast -fno-math-errno -fno-trapping-math \
    -Wall -Wextra solver.cpp -o "$target"
mv -f "$target" famexp
