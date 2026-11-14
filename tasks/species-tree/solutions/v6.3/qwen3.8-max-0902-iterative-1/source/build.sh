#!/bin/sh
# Offline build: compiles stree.cpp into ./stree (single translation unit, no
# external dependencies beyond a C++11 compiler and libm).
set -u
cd "$(dirname "$0")" || exit 0
SRC=stree.cpp
OUT=stree
for CXX in "${CXX:-}" g++ c++ clang++ ; do
    [ -n "$CXX" ] || continue
    command -v "$CXX" >/dev/null 2>&1 || continue
    if "$CXX" -O2 -std=c++11 -o "$OUT.tmp" "$SRC" -lm 2>build.log; then
        mv -f "$OUT.tmp" "$OUT"
        chmod +x "$OUT"
        rm -f build.log
        echo "built $OUT with $CXX"
        exit 0
    fi
done
rm -f "$OUT.tmp"
# Keep going if a prebuilt binary is already usable, so the entry point still works.
if [ -x "$OUT" ]; then
    echo "compilation failed; using prebuilt $OUT" >&2
    exit 0
fi
echo "build failed" >&2
cat build.log >&2 2>/dev/null
exit 1
