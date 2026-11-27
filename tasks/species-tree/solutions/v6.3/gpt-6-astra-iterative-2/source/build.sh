#!/bin/sh
set -eu
cd "$(dirname "$0")"
${CXX:-c++} -O3 -DNDEBUG -std=c++17 -o infer infer.cpp
