#!/bin/sh
# Nothing to compile: the solver is pure Python on top of NumPy/SciPy.
# This only warms/validates the imports so a broken image fails loudly here.
cd "$(dirname "$0")" || exit 1
exec python3 -c "import numpy, scipy, scipy.sparse, scipy.linalg; import solve"
