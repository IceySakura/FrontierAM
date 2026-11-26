#!/usr/bin/env bash
# polarization-demosaicking-v3 after the feedback split (2026-11-26, results/OPEN_ITEMS.md): once the formal baseline
# on the 116 scored scenes is active, rescore every v6.3 cell from its per-scene results, check the feedback baseline on
# the 16 held-out scenes, run one feedback evaluation as a check, publish, then (after the
# continuum reruns' lane has finished) rerun the 8 iterative cells (v63-lane.sh, at most MAX at a time) with one grading
# loop per model (v63-grade.sh grades polarization only while development is idle, and publishes each cell).
# Stops at the first failed check. Usage: polarization-split-chain.sh [MAX]   (log: var/stores/store/reruns/polarization-split-chain.log)
set -uo pipefail
cd "$(dirname "$0")/../.."
MAX=${1:-3}
LOG=var/stores/store/reruns/polarization-split-chain.log
STORE=var/stores/store-v63
CHECK=/tmp/polarization-feedback-check
SPLIT=tasks/polarization-demosaicking/versions/v3/data/private/feedback-split.json
export FRONTIERAM_RUNTIME=$PWD/.runtime-recal
say() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> $LOG; }

until python3 - <<'PY'
import json, sys
from pathlib import Path
b = Path('var/stores/store-v63/baselines/polarization-demosaicking-v3')
parts = json.loads((b / (b / 'ACTIVE').read_text().strip() / 'baseline.json').read_text())['metrics']['parts']
sys.exit(0 if all(parts[w].get('status') == 'ok' and parts[w].get('expected_cases') == 116 for w in ('weak', 'strong')) else 1)
PY
do sleep 60; done
say '"event": "formal_baseline_116"'

.venv-ops/bin/python scripts/grade/polarization-rescore-116.py >> $LOG 2>&1 || { say '"event": "rescore_failed"'; exit 1; }

# the feedback anchors come from the qualified reference run (scripts/grade/polarization-anchors-116.py): a re-measure
# with development running timed the strong reference out at 1.0009 s on two scenes
python3 - "$SPLIT" <<'PY' || { say '"event": "feedback_baseline_failed"'; exit 1; }
import json, sys
b = json.load(open('var/stores/store-public-v63/public-baselines/polarization-demosaicking-v3.json'))
want = sorted(json.load(open(sys.argv[1]))['feedback'])
ok = all(sorted(r['id'] for r in b[w]) == want and all(r['status'] == 'passed' for r in b[w]) for w in ('weak', 'strong'))
sys.exit(0 if ok else 1)
PY
say '"event": "feedback_baseline_16"'

sub=$(python3 -c "import json;print(json.load(open('tasks/polarization-demosaicking/solutions/v6.3/v3/gpt-6.2-sol-single-1/score.json'))['submission'])")
rm -rf $CHECK && mkdir -p $CHECK/src && tar -xf $STORE/submissions/$sub/source.tar -C $CHECK/src
./bench public-eval-v63 --task polarization-demosaicking-v3 --source $CHECK/src --out $CHECK/out --cap 900 --label "feedback split check" >> $LOG 2>&1
python3 - "$CHECK/out/report.json" <<'PY' >> $LOG || { say '"event": "feedback_check_failed"'; exit 1; }
import json, sys
r = json.load(open(sys.argv[1]))
print(json.dumps({'event': 'feedback_check', 'status': r.get('status'), 'score': r.get('score'), 'total': r.get('total'),
                  'categories': sorted(r.get('categories') or {})}))
sys.exit(0 if r.get('status') == 'ok' and (r.get('total') or {}).get('cases') == 16
         and sorted(r.get('categories') or {}) == ['qiu', 'wen'] else 1)
PY

scripts/rounds/publish-now.sh "polarization on 116 scored scenes" >> $LOG 2>&1
say '"event": "published_116"'

while pgrep -f "v63-lane.sh var/stores/store/reruns/continuum-split-queue.txt" > /dev/null; do sleep 120; done
QUEUE=var/stores/store/reruns/polarization-split-queue.txt
cat > $QUEUE <<'EOF'
gpt-6.2-sol iterative polarization-demosaicking-v3 2
gpt-6-astra iterative polarization-demosaicking-v3 2
gpt-6.1-sol iterative polarization-demosaicking-v3 2
claude-opus-5-5 iterative polarization-demosaicking-v3 2
hy4 iterative polarization-demosaicking-v3 2
qwen4-max iterative polarization-demosaicking-v3 2
qwen3.8-max-0902 iterative polarization-demosaicking-v3 2
minimax-m3.1-flash iterative polarization-demosaicking-v3 2
EOF
scripts/rounds/v63-lane.sh $QUEUE "$MAX" >> $LOG 2>&1 &
sleep 30
while read -r model _ _ n; do
    V63_MODEL=$model V63_ATTEMPT=$n V63_TASKS=polarization-demosaicking-v3 scripts/rounds/v63-grade.sh \
        >> var/stores/store/reruns/polarization-split-grade.log 2>&1 &
done < $QUEUE
say '"event": "reruns_started"'
wait
say '"event": "chain_done"'
