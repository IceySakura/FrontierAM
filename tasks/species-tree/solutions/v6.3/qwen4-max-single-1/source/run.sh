#!/bin/sh
set -e
cd "$(dirname "$0")"
if [ ! -x ./species ]; then
  ./build.sh
fi
exec ./species "$1" "$2"
