#!/bin/sh
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ ! -x "$DIR/bin/conic_solver" ]; then
  mkdir -p "$DIR/bin" 2>/dev/null
  g++ -O3 -std=c++17 -fno-math-errno -o "$DIR/bin/conic_solver" "$DIR/src/solver.cpp" -l:libz.so.1 >&2 || true
fi
exec "$DIR/bin/conic_solver" "$@"
