#!/bin/sh
# Usage: ./run.sh /absolute/input/genes.tre /absolute/output/pred.tre
cd "$(dirname "$0")"
exec ./species_tree "$1" "$2" --nni 200000 --rex 2 --time-guard 115
