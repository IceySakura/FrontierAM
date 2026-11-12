#!/bin/sh
# Usage: ./run.sh /abs/input/genes.tre /abs/output/pred.tre
DIR="$(cd "$(dirname "$0")" && pwd)"
IN="$1"; OUT="$2"
export SEED="${SEED:-1}"
if [ ! -x "$DIR/stree" ]; then
  g++ -O2 -std=c++17 -o "$DIR/stree" "$DIR/stree.cpp" 1>&2
fi
"$DIR/stree" "$IN" "$OUT" && [ -s "$OUT" ] && exit 0
echo "primary binary failed; trying generic build" 1>&2
if [ -x "$DIR/stree_generic" ]; then
  "$DIR/stree_generic" "$IN" "$OUT" && [ -s "$OUT" ] && exit 0
fi
echo "falling back to first gene tree topology" 1>&2
python3 "$DIR/fallback.py" "$IN" "$OUT"
