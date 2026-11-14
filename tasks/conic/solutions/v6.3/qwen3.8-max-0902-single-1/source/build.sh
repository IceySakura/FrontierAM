#!/bin/sh
set -e
cd "$(dirname "$0")"
if [ -f expcone.c ]; then EXPF=expcone.c; else EXPF=expcone_fb.c; fi
Z=/usr/lib/x86_64-linux-gnu/libz.so.1
gcc -O2 -march=x86-64 -fno-math-errno -o solver solver.c npz.c $EXPF \
    -llapack -lblas $Z -lm -lgfortran -lquadmath
