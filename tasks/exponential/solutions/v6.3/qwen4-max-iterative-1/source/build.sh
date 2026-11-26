#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
g++ -O3 -std=c++17 -fPIC -shared -o libkernel.so kernel.cpp
