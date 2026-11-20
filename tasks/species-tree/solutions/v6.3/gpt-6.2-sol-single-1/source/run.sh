#!/bin/sh
set -eu
if [ "$#" -ne 2 ]; then
    printf '%s\n' 'usage: run.sh INPUT OUTPUT' >&2
    exit 2
fi
exec "$(dirname -- "$0")/solver" "$1" "$2"
