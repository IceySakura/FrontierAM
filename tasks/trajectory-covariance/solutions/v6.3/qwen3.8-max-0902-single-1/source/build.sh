#!/bin/sh
# Nothing to build: pure Python 3 + NumPy.  Verify imports and warm the cache.
DIR=$(cd "$(dirname "$0")" && pwd)
python3 -c "import numpy, sys; sys.path.insert(0,'$DIR'); import lrcov; print('numpy', numpy.__version__, 'ok')"
