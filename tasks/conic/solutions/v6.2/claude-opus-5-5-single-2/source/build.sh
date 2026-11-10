#!/bin/sh
# Compile the native LDL^T / sparse kernel used by solver.py (loaded via ctypes).
# Never fails the build: a prebuilt libkkt.so is shipped, and solver.py can
# also compile kkt.c into $TMPDIR at run time if the library is missing.
cd "$(dirname "$0")" || exit 0
if gcc -O3 -fPIC -shared -o libkkt.so.new kkt.c -lm; then
    mv -f libkkt.so.new libkkt.so
else
    rm -f libkkt.so.new
    echo "warning: gcc build failed, using shipped libkkt.so" >&2
fi
exit 0
