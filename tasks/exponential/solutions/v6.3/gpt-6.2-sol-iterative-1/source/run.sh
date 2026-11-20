#!/bin/bash
set -eu
cd "$(dirname "$0")"
if [[ ! -x ./propagate ]]; then
    ./build.sh
fi
exec ./propagate
