#!/bin/sh
set -eu

cd "$(dirname "$0")"
g++ -O3 -DNDEBUG -march=native -std=c++17 \
    -I/usr/include/eigen3 rational.cpp \
    /lib/x86_64-linux-gnu/libcholmod.so.3 -o rational
