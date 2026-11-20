#!/bin/sh
set -eu
cd "$(dirname "$0")"
tmp=./solve.build.$$
trap 'rm -f "$tmp"' EXIT
g++ -O3 -std=c++17 -march=znver2 -ffp-contract=fast -fno-math-errno -DNDEBUG solve.cpp -o "$tmp"
mv -f "$tmp" ./solve
