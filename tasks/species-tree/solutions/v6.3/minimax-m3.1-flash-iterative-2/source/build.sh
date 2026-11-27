#!/bin/sh
# No compilation is required: the solution is pure Python 3 (standard library
# only).  This script just byte-compiles the sources so that the first run of
# every instance does not pay the compilation cost.
set -e
dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 -m compileall -q "$dir" >/dev/null 2>&1 || true
exit 0
