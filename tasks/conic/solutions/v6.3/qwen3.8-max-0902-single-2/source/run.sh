#!/bin/sh
d=$(dirname "$0")
cd "$d" || exit 1
if [ ! -f libcore.so ]; then
  sh ./build.sh || exit 1
fi
exec python3 ./solve.py "$@"
