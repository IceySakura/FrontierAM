#!/bin/sh
cd "$(dirname "$0")" || exit 1
if grep -qw avx2 /proc/cpuinfo && grep -qw fma /proc/cpuinfo; then
    exec ./expokit_avx2
fi
exec ./expokit_generic
