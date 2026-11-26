#!/bin/sh
# No build step is required: the estimator only needs Python 3 and NumPy,
# which are already present in the evaluation image.  This script only
# performs a quick self-check.
set -e
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 -c "import numpy, sys; sys.stderr.write('numpy %s, python %s\n' % (numpy.__version__, sys.version.split()[0]))"
python3 -c "import sys; sys.path.insert(0, '$DIR'); import lrv, main; print('build ok')"
