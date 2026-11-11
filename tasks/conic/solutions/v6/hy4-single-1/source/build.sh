#!/bin/sh
# Pure-Python submission: no compilation or dependency installation required.
# (HiGHS is bundled with the preinstalled SciPy wheel.)
set -e
python3 -c "import numpy, scipy, scipy.optimize._highspy._core"
exit 0
