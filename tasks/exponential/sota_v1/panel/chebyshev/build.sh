#!/bin/sh
set -eu
cd "$(dirname "$0")"

common="-std=c99 -Wall -Wextra -Wpedantic -DNDEBUG"
gcc $common -O3 -march=x86-64-v3 -ffp-contract=fast \
    src/stream_chebyshev.c -lm -o chebyshev_avx2
gcc $common -O2 -march=x86-64 \
    src/stream_chebyshev.c -lm -o chebyshev_generic
