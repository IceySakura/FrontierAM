#!/bin/sh
set -eu
cd "$(dirname "$0")"
sh build-tree.sh
mv modified solve
