#!/bin/sh
# Build the solver.  Single threaded C99, no external dependencies.
set -e
cd "$(dirname "$0")"

CC=${CC:-cc}
CFLAGS="-O3 -fno-math-errno -funroll-loops -std=c99 -Wall"

# The solver is written in plain C99 and uses only libm, so it runs on any
# x86-64 (and other) machine without run-time dispatch.
$CC $CFLAGS -DORDER=3 -DLEAF_MAX=256 -o solve solve.c -lm

chmod +x solve
