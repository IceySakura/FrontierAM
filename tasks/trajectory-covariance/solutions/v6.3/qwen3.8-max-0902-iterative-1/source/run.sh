#!/bin/sh
# ./run.sh INPUT OUTPUT   ->  writes the (d,d) long-run covariance estimate.
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$DIR/lrc.py" "$1" "$2"
