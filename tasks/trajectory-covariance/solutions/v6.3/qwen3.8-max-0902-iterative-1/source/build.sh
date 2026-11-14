#!/bin/sh
# Nothing to build: pure Python 3 + NumPy. Byte-compile for faster startup.
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 -m compileall -q "$DIR" >/dev/null 2>&1 || true
exit 0
