#!/bin/sh
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ ! -x "$DIR/infer" ]; then
  "$DIR/build.sh"
fi
exec "$DIR/infer" "$1" "$2"
