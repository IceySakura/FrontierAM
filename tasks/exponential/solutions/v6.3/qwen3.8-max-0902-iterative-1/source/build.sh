#!/bin/bash
# Offline build: compiles the C hot loops into a shared library used by solve.py.
# Falls back silently if no compiler is available (solve.py has a pure-numpy path).
set -u
cd "$(dirname "$0")"
CC_BIN="${CC:-}"
for c in "$CC_BIN" gcc cc clang; do
  [ -z "$c" ] && continue
  command -v "$c" >/dev/null 2>&1 || continue
  if "$c" -O3 -mtune=native -fPIC -shared -o libfastcore.so.tmp fastcore.c -lm 2>build.log; then
    mv -f libfastcore.so.tmp libfastcore.so
    rm -f build.log
    exit 0
  elif "$c" -O3 -fPIC -shared -o libfastcore.so.tmp fastcore.c -lm 2>>build.log; then
    mv -f libfastcore.so.tmp libfastcore.so
    exit 0
  fi
done
rm -f libfastcore.so.tmp
exit 0
