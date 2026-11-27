#!/bin/sh
# Offline build: single C translation unit, no external dependencies.
set -e
cd "$(dirname "$0")"
CC_BIN="${CC:-cc}"
command -v "$CC_BIN" >/dev/null 2>&1 || CC_BIN=gcc
"$CC_BIN" -O3 -std=gnu11 -fomit-frame-pointer -o solver solver.c -lm
