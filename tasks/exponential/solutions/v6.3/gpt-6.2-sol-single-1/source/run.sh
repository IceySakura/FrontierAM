#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x ./famexp ]; then
    ./build.sh
fi
exec ./famexp
