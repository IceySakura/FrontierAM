#!/bin/sh
# Build both TREE-QMC references once into /out/refs-built (same image as grading).
set -e
mkdir -p /out/refs-built
for v in unweighted weighted; do rm -rf /out/refs-built/$v; cp -r /refs/$v /out/refs-built/$v; (cd /out/refs-built/$v && ./build.sh > build.log 2>&1); done
echo built
