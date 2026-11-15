#!/bin/sh
set -e
cd "$(dirname "$0")"
ZLIB=""
for cand in /usr/lib/x86_64-linux-gnu/libz.so.1 /usr/lib/x86_64-linux-gnu/libz.so /lib/x86_64-linux-gnu/libz.so.1 /usr/lib/libz.so.1; do
  if [ -f "$cand" ]; then ZLIB="$cand"; break; fi
done
if [ -z "$ZLIB" ]; then echo "libz not found" >&2; exit 1; fi
CC=${CC:-gcc}
$CC -O2 -fno-math-errno -fPIC -o solver solver.c npz.c sparse.c \
    -llapack -lblas -lgfortran -lm "$ZLIB"
chmod +x run.sh
