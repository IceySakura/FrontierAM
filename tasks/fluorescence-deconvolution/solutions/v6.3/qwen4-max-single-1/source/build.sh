#!/bin/sh
set -u
cd "$(dirname "$0")"
if command -v gcc >/dev/null 2>&1; then
    if gcc -O3 -shared -fPIC -o solver.so.tmp solver.c -lm; then
        mv -f solver.so.tmp solver.so
    else
        rm -f solver.so.tmp
    fi
fi
exit 0
