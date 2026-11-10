#!/bin/sh
# Launch the compiled solver (AVX2/FMA build when the CPU supports it).
cd "$(dirname "$0")" || exit 1
if grep -q ' avx2 ' /proc/cpuinfo 2>/dev/null && grep -q ' fma ' /proc/cpuinfo 2>/dev/null && [ -x ./bin/expprop_avx2 ]; then
    exec ./bin/expprop_avx2
fi
exec ./bin/expprop_generic
