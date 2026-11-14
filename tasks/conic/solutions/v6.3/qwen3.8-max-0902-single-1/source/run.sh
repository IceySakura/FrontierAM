#!/bin/sh
cd "$(dirname "$0")"
exec ./solver --input "$2" --output "$4"
