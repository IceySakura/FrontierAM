#!/bin/sh
set -eu
dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ ! -x "$dir/species_tree" ]; then
    (cd "$dir" && ./build.sh)
fi
exec "$dir/species_tree" "$1" "$2"
