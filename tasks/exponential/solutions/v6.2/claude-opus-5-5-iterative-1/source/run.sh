#!/bin/sh
# Launch the compiled solver.  Fallbacks (all before READY, so untimed):
# copy the shipped binary to /tmp if it lost its exec bit, or rebuild it with
# the system gcc if it is missing or the CPU lacks AVX2/FMA.
DIR=$(cd "$(dirname "$0")" && pwd)
BIN="$DIR/expo"
CPU_OK=0
if grep -q avx2 /proc/cpuinfo 2>/dev/null && grep -q fma /proc/cpuinfo 2>/dev/null; then CPU_OK=1; fi
if [ "$CPU_OK" = 1 ] && [ -f "$BIN" ]; then
    if [ -x "$BIN" ]; then
        exec "$BIN"
    fi
    TMPB=$(mktemp -d /tmp/expo.XXXXXX) && cp "$BIN" "$TMPB/expo" && chmod +x "$TMPB/expo" && exec "$TMPB/expo"
fi
TMPB=$(mktemp -d /tmp/expo.XXXXXX)
if gcc -O3 -march=native -o "$TMPB/expo" "$DIR/expo.c" -lm 2>/dev/null || gcc -O2 -o "$TMPB/expo" "$DIR/expo.c" -lm; then
    exec "$TMPB/expo"
fi
exit 1
