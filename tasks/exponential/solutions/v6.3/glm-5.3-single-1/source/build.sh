#!/bin/sh
# Offline build: precompile the C kernels so the runtime never needs a compiler.
set -e
cd "$(dirname "$0")"
CC=${CC:-cc}
for flags in "-O3" "-O2"; do
    if $CC $flags -shared -fPIC -o famexp_kernel.so kernel.c -lm 2>/dev/null; then
        exit 0
    fi
done
exit 0
