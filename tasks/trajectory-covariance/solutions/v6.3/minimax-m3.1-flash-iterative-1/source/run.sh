#!/bin/sh
# Usage: ./run.sh INPUT.npy OUTPUT.npy
set -e
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$DIR/main.py" "$1" "$2"
