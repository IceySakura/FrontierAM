#!/bin/bash
# Rebuilds the two prebuilt binaries shipped with the submission (run during
# development; run.sh also falls back to compiling from source if needed).
set -e
cd "$(dirname "$0")"
CFLAGS="-O3 -fno-math-errno -Wall -Wno-unused-function -Wno-maybe-uninitialized"
gcc $CFLAGS -march=x86-64-v3 -mtune=znver2 -o expo_v3 expo.c -lm
gcc $CFLAGS -march=x86-64 -mtune=generic -o expo_v1 expo.c -lm
