#!/usr/bin/env bash
# blackbox-cubature v3.6 dual trials (tasks/blackbox-cubature/analysis/DUAL_PLAN.md 9.18): grade each attempt as soon
# as its log (scripts/rounds/v3-dual-trial.sh) shows develop_finished. The frozen source and the proxy log go to
# tasks/blackbox-cubature/solutions/v3.6-trial/<model>-iterative-<n>/, the theory gets the Lean check
# (scripts/grade/lean-comparator.py on a pinned grade lease), the program a 4-shard private suite scored against the v3.6
# reference suite (scripts/grade/cubature-suite-score.py: three-track score and the two-track G). The theory review
# against analysis/THEORY_REVIEW.md stays manual.
# Usage: v36-dual-grade.sh REFERENCE_SUITE [TAG ...]   (tags sol62, opus; default both)
set -uo pipefail
cd "$(dirname "$0")/../.."
R=$PWD
REF=$1
shift
TAGS=${*:-sol62 opus}
STORE=var/stores/store-v63
RT=$R/.runtime-recal
field() { python3 -c "import json,sys; print(json.loads(sys.stdin.read())['$1'])"; }

grade() {
  local tag=$1 model log out line sub run suite
  case $tag in sol62) model=gpt-6.2-sol;; opus) model=claude-opus-5-5;; *) echo "unknown tag $tag" >&2; return 1;; esac
  log=var/stores/store/reruns/v36trial-$tag-iterative-1.log
  out=tasks/blackbox-cubature/solutions/v3.6-trial/$model-iterative-1
  until grep -q '"event": "develop_finished"' "$log" 2>/dev/null; do sleep 60; done
  line=$(grep '"event": "develop_finished"' "$log" | tail -1)
  sub=$(echo "$line" | field submission)
  run=$(echo "$line" | field run)
  mkdir -p "$out"
  python3 - "$STORE/submissions/$sub/source.tar" "$out/source" <<'PY'
import sys, tarfile
with tarfile.open(sys.argv[1]) as tar:
    tar.extractall(sys.argv[2], filter='data')
PY
  if [ -f "$STORE/dev-runs/$run/netproxy.jsonl" ]; then cp "$STORE/dev-runs/$run/netproxy.jsonl" "$out/"; fi
  PYTHONPATH=$R python3 scripts/ops/with_lease.py --kind grade --pin --label "lean check v3.6 $tag" --timeout 14400 -- \
      python3 scripts/grade/lean-comparator.py "$out/source" "$out/lean-check" --timeout 5400 > "$out/lean-check.out" 2>&1 &
  suite=$(FRONTIERAM_RUNTIME=$RT ./bench --store $STORE suite --task blackbox-cubature-v3.6 --submission "$sub" \
      --shards 4 --isolation shared --cap-s 3600 2>&1 | tail -1 | field id)
  echo "{\"submission\": \"$sub\", \"run\": \"$run\", \"suite\": \"$suite\"}" > "$out/frozen.json"
  FRONTIERAM_RUNTIME=$RT ./bench --store $STORE grade-worker --task blackbox-cubature-v3.6 --workers 4 --drain \
      > "$out/grade-worker.log" 2>&1
  # another drain worker may hold some shards: wait until every shard has metrics (at most 4 h)
  for _ in $(seq 240); do
    python3 - "$STORE" "$suite" <<'PY' && break
import json, sys
from pathlib import Path
store, suite = Path(sys.argv[1]), sys.argv[2]
shards = json.loads((store / 'suites' / f'{suite}.json').read_text())['shards']
sys.exit(0 if all((store / 'results' / s['job'] / 'grade/metrics.json').exists() for s in shards) else 1)
PY
    sleep 60
  done
  .venv-ops/bin/python scripts/grade/cubature-suite-score.py --store $STORE --task blackbox-cubature-v3.6 \
      --reference "$REF" --candidate "$suite" --out "$out/score.json" > /dev/null
  wait
}

for tag in $TAGS; do grade "$tag" & done
wait
