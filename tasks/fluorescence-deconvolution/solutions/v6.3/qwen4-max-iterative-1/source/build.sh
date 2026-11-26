#!/bin/sh
set -e
cd "$(dirname "$0")"
if command -v gcc >/dev/null 2>&1; then
    gcc -O3 -march=native -ffast-math -shared -fPIC deconv.c -o deconv.so.tmp -lm && mv deconv.so.tmp deconv.so || true
fi
exit 0
