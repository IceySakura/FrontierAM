#!/bin/sh
set -eu
dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ ! -x "$dir/species_tree" ]; then
    (cd "$dir" && sh ./build.sh) >&2
fi
exec "$dir/species_tree" "$1" "$2"
