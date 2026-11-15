#!/bin/sh
# Offline build: single core, produces ./solve next to this script.
set -e
cd "$(dirname "$0")"
CC=${CC:-gcc}
$CC -O3 -mavx2 -mfma -mbmi2 -funroll-loops -fno-math-errno -o solve solve.c -lm
