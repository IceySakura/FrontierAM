#!/bin/sh
set -eu
cd "$(dirname "$0")"
if grep -qw avx2 /proc/cpuinfo && grep -qw fma /proc/cpuinfo; then
    exec ./chebyshev_avx2
fi
exec ./chebyshev_generic
