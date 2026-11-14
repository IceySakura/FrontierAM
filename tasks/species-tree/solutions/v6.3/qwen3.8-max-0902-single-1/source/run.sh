#!/bin/sh
# usage: ./run.sh INPUT OUTPUT
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
exec "$DIR/sp" "$1" "$2"
