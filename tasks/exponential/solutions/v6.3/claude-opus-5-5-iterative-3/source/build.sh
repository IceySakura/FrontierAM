#!/bin/bash
# Builds the native solver binaries (prebuilt copies are shipped as well).
cd "$(dirname "$0")"
CF="-O3 -ffp-contract=off -fno-math-errno -Wall -Wno-unused-result"
LIBS="-llapack -lblas -lm -ldl"
ok=0
gcc $CF -march=x86-64-v3 -mtune=znver2 -o expo_v3.tmp expo_main.c $LIBS && mv expo_v3.tmp expo_v3 && ok=1
gcc $CF -march=x86-64 -mtune=generic -o expo_generic.tmp expo_main.c $LIBS && mv expo_generic.tmp expo_generic && ok=1
# Python fallback kernels
gcc -O3 -fPIC -shared -ffp-contract=off -fno-math-errno -march=x86-64 -o libexpo_generic.so expo.c -lm || true
gcc -O3 -fPIC -shared -ffp-contract=off -fno-math-errno -march=x86-64-v3 -o libexpo_v3.so expo.c -lm || true
[ $ok = 1 ]
