#!/bin/sh
# Compile the small C helper library used for sparse matvecs and Schur
# complement assembly.  Falls back silently: solver_core.py has pure numpy
# implementations of every routine.
set -e
cd "$(dirname "$0")"
CC="${CC:-cc}"
if ! command -v "$CC" >/dev/null 2>&1; then
  for c in cc gcc clang; do
    if command -v "$c" >/dev/null 2>&1; then CC="$c"; break; fi
  done
fi
"$CC" -O2 -fPIC -shared -o libfast.so fast.c -lm
