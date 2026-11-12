#!/bin/sh
set -u
cd "$(dirname "$0")" || exit 0
if ! "${CC:-cc}" -O3 -std=c99 -march=native -ffp-contract=fast -fPIC -shared cg_kernels.c -o libcg.so; then
    rm -f libcg.so
fi
exit 0
