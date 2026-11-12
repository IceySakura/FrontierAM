#!/bin/sh
set -e
cd "$(dirname "$0")"
CC=${CC:-gcc}
OPTS="${FMM_OPTS:-}"
$CC -O3 -mavx2 -mfma $OPTS -c fmm.c -o fmm_avx2.o
$CC -O3 $OPTS -c fmm.c -o fmm_gen.o
$CC -O3 -c main.c -o main.o
$CC -O3 -o solve main.o fmm_avx2.o fmm_gen.o -lm
