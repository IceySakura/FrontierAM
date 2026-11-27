#!/bin/sh
# ./run.sh INPUT OUTPUT
set -e
d=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$d/solver" "$1" "$2"
