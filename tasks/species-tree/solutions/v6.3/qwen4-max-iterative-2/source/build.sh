#!/bin/sh
set -eu
cd "$(dirname "$0")"
g++ -O3 -std=c++17 -o species species.cpp
