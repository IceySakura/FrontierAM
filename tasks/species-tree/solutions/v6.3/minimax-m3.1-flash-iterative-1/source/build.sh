#!/bin/sh
# Nothing to compile: the solution is a single Python 3 program.
# This script only verifies that the interpreter and NumPy are usable.
set -e
python3 -c "import sys; print('python', sys.version.split()[0])"
python3 -c "import numpy; print('numpy', numpy.__version__)" 2>/dev/null || \
    echo "numpy not available - pure python fallback will be used"
exit 0
