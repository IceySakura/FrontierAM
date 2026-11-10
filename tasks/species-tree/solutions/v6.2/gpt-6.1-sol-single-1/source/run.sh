#!/bin/sh
set -eu
if [ "$#" -ne 2 ]; then
    printf '%s\n' "Usage: ./run.sh INPUT OUTPUT" >&2
    exit 2
fi
base=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ ! -x "$base/infer" ]; then
    "$base/build.sh"
fi
exec "$base/infer" "$1" "$2"
