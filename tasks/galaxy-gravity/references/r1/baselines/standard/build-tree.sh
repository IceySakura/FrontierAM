#!/bin/sh
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
tar -xzf "$here/Treecode2-16051d8.tar.gz" -C "$build"
cd "$build"
make -s -C clib ZCC="${CC:-cc}" libClib.a
for mode in standard modified; do
    if [ "$mode" = standard ]; then grav=treegrav.c; set --; else grav=treegrav_mod.c; set -- -DMODTREECODE; fi
    "${CC:-cc}" -O3 -std=gnu17 -DLINUX -DDOUBLEPREC -fno-strict-aliasing "$@" \
        -I. "$here/adapter.c" treebuild.c "$grav" clib/libClib.a -lm -o "$here/$mode"
done
