#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec ./solver "$1" "$2"
