#!/bin/bash
# Rebuild the solver binaries from source. Prebuilt binaries are shipped in bin/;
# they are only replaced when the compile succeeds, so a failed build keeps them.
cd "$(dirname "$0")" || exit 0
mkdir -p bin
if gcc -O3 -mavx2 -mfma -o bin/expprop_avx2.new expprop.c -lm 2>/dev/null; then
    mv -f bin/expprop_avx2.new bin/expprop_avx2
fi
if gcc -O3 -o bin/expprop_generic.new expprop.c -lm 2>/dev/null; then
    mv -f bin/expprop_generic.new bin/expprop_generic
fi
rm -f bin/*.new
exit 0
