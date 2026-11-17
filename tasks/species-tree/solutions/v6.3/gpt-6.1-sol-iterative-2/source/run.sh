#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec ./infer "$1" "$2"
