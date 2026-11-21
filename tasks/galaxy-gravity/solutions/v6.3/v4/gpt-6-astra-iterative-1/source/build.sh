#!/bin/sh
set -eu
cd "$(dirname "$0")"
python3 generate.py
python3 generate_harmonic.py
exec g++ -O3 -std=c++17 -mavx2 -mfma -fno-math-errno -ffp-contract=fast -DNDEBUG main.cpp -o solve
