#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec gcc -O3 -ffast-math -march=native -fPIC -shared core.c -o core.so
