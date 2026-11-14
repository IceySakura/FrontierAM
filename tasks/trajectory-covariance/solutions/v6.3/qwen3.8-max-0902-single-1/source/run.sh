#!/bin/sh
# ./run.sh INPUT OUTPUT   (single trajectory -> long-run covariance matrix)
DIR=$(cd "$(dirname "$0")" && pwd)
exec python3 "$DIR/main.py" "$1" "$2"
