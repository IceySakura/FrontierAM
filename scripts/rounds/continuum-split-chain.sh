#!/usr/bin/env bash
# continuum-estimation after the feedback split (2026-11-26, results/OPEN_ITEMS.md). Once the formal baseline measured on
# the 45 scored cases is active: measure the feedback baseline on the 15 held-out cases, run one feedback evaluation end
# to end as a check, regrade every v6.3 cell on the 45 cases (regrade-continuum45.sh, which publishes), then rerun the 8
# iterative cells (v63-lane.sh, at most MAX at a time) with one grading loop per model (v63-grade.sh publishes each cell).
# Stops at the first failed check. Usage: continuum-split-chain.sh [MAX]   (log: var/stores/store/reruns/continuum-split-chain.log)
set -uo pipefail
cd "$(dirname "$0")/../.."
MAX=${1:-3}
LOG=var/stores/store/reruns/continuum-split-chain.log
STORE=var/stores/store-v63
CHECK=/tmp/continuum-feedback-check
export FRONTIERAM_RUNTIME=$PWD/.runtime-recal
say() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> $LOG; }

until python3 - <<'PY'
import json, sys
from pathlib import Path
b = Path('var/stores/store-v63/baselines/continuum-estimation')
parts = json.loads((b / (b / 'ACTIVE').read_text().strip() / 'baseline.json').read_text())['metrics']['parts']
sys.exit(0 if all(parts[w].get('status') == 'ok' and parts[w].get('cases') == 45 for w in ('weak', 'strong')) else 1)
PY
do sleep 60; done
say '"event": "formal_baseline_45"'

./bench public-baseline-v63 --task continuum-estimation --replace >> $LOG 2>&1
python3 -c "
import json, sys
b = json.load(open('var/stores/store-public-v63/public-baselines/continuum-estimation.json'))
sys.exit(0 if len(b['weak']) == len(b['strong']) == 15 else 1)" || { say '"event": "feedback_baseline_failed"'; exit 1; }
say '"event": "feedback_baseline_15"'

sub=$(python3 -c "import json;print(json.load(open('tasks/continuum-estimation/solutions/v6.3/gpt-6.2-sol-single-1/score.json'))['submission'])")
rm -rf $CHECK && mkdir -p $CHECK/src && tar -xf $STORE/submissions/$sub/source.tar -C $CHECK/src
./bench public-eval-v63 --task continuum-estimation --source $CHECK/src --out $CHECK/out --cap 900 --label "feedback split check" >> $LOG 2>&1
python3 - "$CHECK/out/report.json" <<'PY' >> $LOG || { say '"event": "feedback_check_failed"'; exit 1; }
import json, sys
r = json.load(open(sys.argv[1]))
print(json.dumps({'event': 'feedback_check', 'status': r.get('status'), 'score': r.get('score'), 'total': r.get('total')}))
sys.exit(0 if r.get('status') == 'ok' and r.get('score') is not None and (r.get('total') or {}).get('cases') == 15 else 1)
PY

scripts/rounds/regrade-continuum45.sh >> $LOG 2>&1
say '"event": "regraded_45"'

QUEUE=var/stores/store/reruns/continuum-split-queue.txt
cat > $QUEUE <<'EOF'
gpt-6.2-sol iterative continuum-estimation 2
gpt-6-astra iterative continuum-estimation 2
gpt-6.1-sol iterative continuum-estimation 2
claude-opus-5-5 iterative continuum-estimation 2
hy4 iterative continuum-estimation 2
qwen4-max iterative continuum-estimation 2
qwen3.8-max-0902 iterative continuum-estimation 3
minimax-m3.1-flash iterative continuum-estimation 2
EOF
scripts/rounds/v63-lane.sh $QUEUE "$MAX" >> $LOG 2>&1 &
sleep 30
while read -r model _ _ n; do
    V63_MODEL=$model V63_ATTEMPT=$n V63_TASKS=continuum-estimation scripts/rounds/v63-grade.sh \
        >> var/stores/store/reruns/continuum-split-grade.log 2>&1 &
done < $QUEUE
say '"event": "reruns_started"'
wait
say '"event": "chain_done"'
