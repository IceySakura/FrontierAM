#!/bin/sh
exec python3 "$(dirname "$0")/species_tree.py" "$1" "$2"
