#!/bin/sh
# Usage: ./run.sh INPUT OUTPUT
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
exec python3 "$DIR/src/solve.py" "$1" "$2"
