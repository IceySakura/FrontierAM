#!/bin/sh
# Usage: ./run.sh INPUT OUTPUT
set -e
DIR=$(dirname "$0")
exec python3 "$DIR/species_tree.py" "$1" "$2"
