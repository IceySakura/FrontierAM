#!/bin/sh
# Usage: ./run.sh /abs/input/genes.tre /abs/output/pred.tre
DIR="$(cd "$(dirname "$0")" && pwd)"
IN="$1"; OUT="$2"
[ -n "$SEED" ] || SEED=1; export SEED
if [ ! -x "$DIR/stree_generic" ]; then (cd "$DIR" && sh ./build.sh) >&2; fi
BIN="$DIR/stree_generic"
if grep -qw avx2 /proc/cpuinfo 2>/dev/null && grep -qw fma /proc/cpuinfo 2>/dev/null && [ -x "$DIR/stree_avx2" ]; then BIN="$DIR/stree_avx2"; fi
valid() { [ -s "$OUT" ] && grep -q ';' "$OUT"; }
rm -f "$OUT"
timeout -k 3 95 "$BIN" "$IN" "$OUT" >&2
st=$?
if [ $st -ne 0 ] && ! valid; then
  echo "primary binary failed ($st); trying generic" >&2
  STREE_BOOT=20 STREE_TLIM=8 timeout -k 3 25 "$DIR/stree_generic" "$IN" "$OUT" >&2
fi
if ! valid; then
  echo "falling back to caterpillar tree" >&2
  python3 - "$IN" "$OUT" << 'PY'
import re, sys
s = open(sys.argv[1]).read().split(';')[0]
labs = [m for m in re.findall(r'[(,]\s*([^():,;\s]+)', s)]
t = labs[0]
for x in labs[1:]: t = '(%s,%s)' % (t, x)
open(sys.argv[2], 'w').write(t + ';\n')
PY
fi
exit 0
