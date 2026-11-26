#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -f libkernel.so ]]; then
  ./build.sh
fi
exec python3 solver.py
