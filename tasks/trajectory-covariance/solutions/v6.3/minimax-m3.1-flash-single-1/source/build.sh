#!/bin/sh
# Nothing has to be compiled: the solution is pure Python 3 + NumPy.
# This script only verifies that the runtime is available.
set -e
python3 - <<'PY'
import sys
import numpy
print("python", sys.version.split()[0], "numpy", numpy.__version__)
PY
