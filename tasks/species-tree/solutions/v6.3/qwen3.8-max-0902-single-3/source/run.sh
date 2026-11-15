#!/bin/sh
# Run the species-tree inference program.
# usage: ./run.sh INPUT OUTPUT
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
if [ ! -x "$DIR/species" ]; then
  if [ -x "$DIR/build.sh" ]; then "$DIR/build.sh"; fi
fi
exec "$DIR/species" "$1" "$2"
