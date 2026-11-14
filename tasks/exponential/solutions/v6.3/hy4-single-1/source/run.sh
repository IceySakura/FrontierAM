#!/bin/sh
set -eu
cd "$(dirname "$0")"
sig="$(g++ -dumpmachine)-$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null | sha256sum | cut -d' ' -f1)"
if [ ! -x ./solve ] || [ ! -f .buildstamp ] || [ "$(cat .buildstamp 2>/dev/null)" != "$sig" ]; then
    sh ./build.sh
fi
exec ./solve
