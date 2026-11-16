#!/bin/bash
# Rebuilds the solver executables and the ctypes kernel library from source.
# Prebuilt copies ship with the submission; each is replaced only if gcc succeeds.
cd "$(dirname "$0")" || exit 0
build() {
    local out=$1; shift
    if gcc -O3 "$@" -o "$out.tmp" -lm 2>/dev/null; then
        mv -f "$out.tmp" "$out"
    else
        rm -f "$out.tmp"
    fi
}
V3="-march=x86-64-v3 -mtune=znver2 -fno-tree-vectorize"
build expsolve_v3 $V3 expsolve.c
build expsolve_gen -mtune=generic expsolve.c
build libexpkern_v3.so -shared -fPIC $V3 expkern.c
build libexpkern_gen.so -shared -fPIC -mtune=generic expkern.c
exit 0
