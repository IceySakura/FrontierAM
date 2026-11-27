#!/bin/sh
set -eu
cd "$(dirname "$0")"
${CXX:-g++} -O3 -std=c++17 -DNDEBUG -Wall -Wextra infer.cpp -o infer
