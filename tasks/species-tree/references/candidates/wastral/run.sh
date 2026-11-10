#!/bin/sh
# Weighted ASTRAL, hybrid weighting (default mode) with local Bayesian supports (-B: support range 1/3 to 1), one thread,
# topology only.
exec "$(dirname "$0")/wastral" -B -u 0 -t 1 -o "$2" "$1" 2>/dev/null
