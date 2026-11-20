#!/bin/sh
set -eu
cd "$(dirname "$0")"
if ! cc -O3 -march=native -mavx2 -mfma -fPIC -shared -o cg_native.so cg_native.c; then
    rm -f cg_native.so
fi
