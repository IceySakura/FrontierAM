#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
g++ -std=c++17 -O3 -fPIC -shared "$HERE/ldl.cpp" -o "$HERE/ldl.so"
python -m compileall -q "$HERE"
