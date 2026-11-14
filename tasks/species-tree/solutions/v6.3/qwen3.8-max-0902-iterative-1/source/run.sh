#!/bin/sh
# ./run.sh INPUT OUTPUT   -- infer one species tree from the gene trees in INPUT
set -u
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BIN="$DIR/stree"
IN=${1:-}
OUT=${2:-}
if [ -z "$IN" ] || [ -z "$OUT" ]; then
    echo "usage: run.sh INPUT OUTPUT" >&2
    exit 2
fi
if [ ! -x "$BIN" ]; then
    sh "$DIR/build.sh" >&2 || true
fi
if [ -x "$BIN" ]; then
    exec "$BIN" "$IN" "$OUT"
fi
echo "no usable binary" >&2
exit 1
