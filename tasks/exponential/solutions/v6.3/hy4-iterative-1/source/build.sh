#!/bin/sh
# Offline build of the Chebyshev kernel.  Only gcc and libc are required.
set -e
cd "$(dirname "$0")"
gcc -O3 -fPIC -shared -fno-math-errno -funroll-loops \
    -o libfamexp.so kernel.c -lm
