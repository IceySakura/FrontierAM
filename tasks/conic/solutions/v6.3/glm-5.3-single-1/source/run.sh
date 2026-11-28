#!/bin/sh
# Conic solver entry point.  Requires --input and --output (absolute paths).
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$DIR/conic.py" "$@"
