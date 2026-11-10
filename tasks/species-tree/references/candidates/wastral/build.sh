#!/bin/sh
set -e
cd "$(dirname "$0")"
g++ -std=gnu++11 -march=native -Ofast src/astral-hybrid.cpp -o wastral -pthread || g++ -std=c++17 -O2 src/astral-hybrid.cpp -o wastral -pthread
