#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec ./species "$1" "$2" 48 2 8
