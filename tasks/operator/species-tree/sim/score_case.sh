#!/bin/sh
# Score one generated case with both prebuilt references: /out/<case>/refs.json (normalised RF and wall seconds).
set -e
case=$1
for v in unweighted weighted; do
  out=/tmp/pred-$v.tre; rm -f $out
  t0=$(date +%s.%N); SEED=1 /out/refs-built/$v/run.sh /out/$case/genes.tre $out > /dev/null 2>&1; t1=$(date +%s.%N)
  python3 -c "
import json, sys; sys.path.insert(0, '/sim'); from pipeline import nrf
print(json.dumps({'reference': '$v', 'nrf': nrf(open('/out/$case/truth.tre').read(), open('$out').read()), 'seconds': round($t1 - $t0, 1)}))" >> /out/$case/refs.jsonl
done
