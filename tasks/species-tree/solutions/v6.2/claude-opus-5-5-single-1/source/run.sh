#!/bin/sh
# Usage: ./run.sh /abs/input/genes.tre /abs/output/pred.tre
DIR="$(cd "$(dirname "$0")" && pwd)"
IN="$1"; OUT="$2"
if [ ! -x "$DIR/stree_base" ]; then
  (cd "$DIR" && sh ./build.sh) >&2
fi
BIN="$DIR/stree_base"
if [ -x "$DIR/stree_avx2" ] && grep -q avx2 /proc/cpuinfo 2>/dev/null && grep -q fma /proc/cpuinfo 2>/dev/null; then
  BIN="$DIR/stree_avx2"
fi
export SEED="${SEED:-1}"
start=$(date +%s 2>/dev/null || echo 0)
"$BIN" "$IN" "$OUT"
st=$?
if [ $st -ne 0 ] && [ "$BIN" != "$DIR/stree_base" ]; then
  now=$(date +%s 2>/dev/null || echo 0)
  if [ $((now - start)) -lt 20 ]; then
    # optimized binary failed early (e.g. unsupported instruction): portable fallback
    STREE_EMERG=80 "$DIR/stree_base" "$IN" "$OUT"
    st=$?
  fi
fi
exit $st
