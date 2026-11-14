#!/bin/sh
set -eu
cd "$(dirname "$0")"
# Offline build: the frozen image provides g++ and libm.  The stamp makes
# run.sh rebuild a native binary if the clean copy is executed on another host.
sig="$(g++ -dumpmachine)-$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null | sha256sum | cut -d' ' -f1)"
g++ -O3 -std=c++17 -march=native -mtune=native -fno-fast-math -o solve solve.cpp -lm
printf '%s\n' "$sig" > .buildstamp
