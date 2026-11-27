#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec ./infer "$1" "$2" --starts 1 --distance 3 --max-support 8 \
    --distance-prior 0.5 --distance-zero 0.00001
