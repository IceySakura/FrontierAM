#!/bin/bash
# Compile the native kernels (generic x86-64 and AVX2/FMA variants).
# -ffp-contract=off keeps the fused and single Lanczos kernels bitwise identical.
cd "$(dirname "$0")"
mkdir -p build
gcc -O3 -ffp-contract=off -shared -fPIC -o build/libk_base.so kernels.c -lm || true
gcc -O3 -ffp-contract=off -mavx2 -mfma -shared -fPIC -o build/libk_avx2.so kernels.c -lm || true
exit 0
