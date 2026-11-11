#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x ./solver ]; then
    sh ./build.sh
fi
exec ./solver
