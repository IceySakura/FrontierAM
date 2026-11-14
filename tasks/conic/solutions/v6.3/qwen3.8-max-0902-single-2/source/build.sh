#!/bin/sh
set -e
cd "$(dirname "$0")"
CC=${CC:-gcc}
if ! $CC -O3 -march=native -fPIC -shared -o libcore.so core.c -lm 2>build.log; then
  if ! $CC -O2 -fPIC -shared -o libcore.so core.c -lm 2>>build.log; then
    cat build.log
    exit 1
  fi
fi
exit 0
