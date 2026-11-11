#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec ./infer "$1" "$2" --lambda 0.1 --absolute-lambda 2 --relative --normalize --pendant --balance 0.25 --time 125
