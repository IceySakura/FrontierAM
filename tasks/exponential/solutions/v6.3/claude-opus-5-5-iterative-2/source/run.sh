#!/bin/sh
# Native solver: AVX2/FMA build when the CPU supports it (CPUID check by the generic build,
# or /proc/cpuinfo), otherwise the generic build; Python fallback if neither can run.
cd "$(dirname "$0")"
if [ -x ./expsolve_v3 ]; then
    if ./expsolve_gen --has-v3 2>/dev/null; then
        exec ./expsolve_v3
    fi
    if grep -q -w avx2 /proc/cpuinfo 2>/dev/null && grep -q -w fma /proc/cpuinfo 2>/dev/null; then
        exec ./expsolve_v3
    fi
fi
if [ -x ./expsolve_gen ]; then
    exec ./expsolve_gen
fi
exec python3 solver.py
