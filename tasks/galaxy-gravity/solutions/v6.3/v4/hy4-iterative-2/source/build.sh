#!/bin/sh
set -e
cd "$(dirname "$0")"
g++ -O3 -mavx2 -mfma -ffp-contract=fast -fno-math-errno -o solve galaxy.cpp -lm
