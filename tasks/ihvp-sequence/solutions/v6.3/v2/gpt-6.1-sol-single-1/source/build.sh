#!/bin/sh
set -eu
cd "$(dirname "$0")"
"${CC:-cc}" -O3 -fPIC -shared -fopenmp-simd -march=native -ffp-contract=fast \
    cg_kernel.c -o _cg.so
