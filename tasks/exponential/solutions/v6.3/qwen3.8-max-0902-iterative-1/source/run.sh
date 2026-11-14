#!/bin/bash
# FAMEXP2 solver entry point.  The compiled hot loop is normally produced by
# build.sh; rebuild it here only if it is missing (before READY, so untimed).
cd "$(dirname "$0")" || exit 1
if [ ! -f ./libfastcore.so ] && [ ! -f /tmp/libfastcore.so ]; then
  for c in gcc cc clang; do
    if command -v "$c" >/dev/null 2>&1; then
      "$c" -O3 -fPIC -shared -o ./libfastcore.so fastcore.c -lm 2>/dev/null && break
      "$c" -O3 -fPIC -shared -o /tmp/libfastcore.so fastcore.c -lm 2>/dev/null && break
    fi
  done
fi
exec python3 ./solve.py
