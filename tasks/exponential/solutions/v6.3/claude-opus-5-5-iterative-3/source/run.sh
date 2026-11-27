#!/bin/bash
# FAMEXP2 solver entry point; the working directory is the submission directory.
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 GOTO_NUM_THREADS=1
BIN=""
if grep -q ' avx2' /proc/cpuinfo 2>/dev/null && grep -q ' fma' /proc/cpuinfo 2>/dev/null && [ -x ./expo_v3 ]; then
  BIN=./expo_v3
elif [ -x ./expo_generic ]; then
  BIN=./expo_generic
fi
if [ -n "$BIN" ] && "$BIN" --selftest >/dev/null 2>&1; then
  exec "$BIN"
fi
# rebuild into a temporary directory if the shipped binaries do not run here
TMPB=$(mktemp -d 2>/dev/null || echo /tmp)
if gcc -O3 -ffp-contract=off -fno-math-errno -o "$TMPB/expo" expo_main.c -llapack -lblas -lm >/dev/null 2>&1 \
   && "$TMPB/expo" --selftest >/dev/null 2>&1; then
  exec "$TMPB/expo"
fi
exec python3 solve.py
