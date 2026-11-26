#!/bin/sh
set -e
cd "$(dirname "$0")"
gcc -O2 -I. ipm.c npzio.c inflate.c kkt.c dense.c -lm -o solver
