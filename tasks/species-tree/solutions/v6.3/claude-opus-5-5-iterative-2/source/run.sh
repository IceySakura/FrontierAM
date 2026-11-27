#!/bin/sh
# usage: ./run.sh /abs/input/genes.tre /abs/output/pred.tre
DIR="$(cd "$(dirname "$0")" && pwd)"
IN="$1"; OUT="$2"
[ -x "$DIR/st" ] || sh "$DIR/build.sh" >&2
export SEED="${SEED:-1}"
# method configuration (same values are the built-in defaults of st; see README)
export SMODE=2 HLAM=0.2 HMIXF=0.2 HNORM=1 QK=3 WEXP=1 XMINP=0.2 NJN=1 NJBOOT=0 NBOOT=0 TLIM=60 HTLIM=60
if ! "$DIR/st" "$IN" "$OUT"; then
  echo "main solver failed; writing fallback tree" >&2
  python3 "$DIR/fallback.py" "$IN" "$OUT"
fi
