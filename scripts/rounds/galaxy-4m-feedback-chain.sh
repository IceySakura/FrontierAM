#!/usr/bin/env bash
# galaxy-gravity-v4: the 4,194,304-particle feedback snapshot fb-4194304-disc (user decision 2026-11-26,
# results/OPEN_ITEMS.md), already in the sealed feedback pool with a provisional 300 s limit. Once no galaxy-gravity-v4
# grading is running: time the reference pair on it (three calls under a uniform generous limit, as
# scripts/analyze/galaxy-v4-calibrate.py does), set its limit by the B.4 rule (1.25 x the strong reference's largest
# CPU time, rounded up to 0.1 s), refresh the sealed feedback pool and workspace, measure the feedback baseline on the 16
# snapshots (one call each), and run one feedback evaluation as a check. Stops at the first failed check.
# Usage: galaxy-4m-feedback-chain.sh [WAIT_SUITE]   (log: var/stores/store/reruns/galaxy-4m-feedback.log)
set -uo pipefail
cd "$(dirname "$0")/../.."
WAIT_SUITE=${1:-}
LOG=var/stores/store/reruns/galaxy-4m-feedback.log
PUB=var/stores/store-public-v63
CHECK=/tmp/galaxy-feedback-check
export FRONTIERAM_RUNTIME=$PWD/.runtime-recal
say() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> $LOG; }
galaxy_busy() {
    nexus/.venv/bin/python - "$WAIT_SUITE" <<'PY'
import json, os, sys
sys.path.insert(0, '.')
from scripts.ops.slot_status import snapshot
leased = any((c.get('lease') or {}).get('task') == 'galaxy-gravity-v4' for c in snapshot()['cores'])
suite, S = sys.argv[1], 'var/stores/store-v63'
waiting = bool(suite) and not all(os.path.exists(f'{S}/jobs/done/{s["job"]}.json')
                                  for s in json.load(open(f'{S}/suites/{suite}.json'))['shards'])
sys.exit(0 if leased or waiting else 1)
PY
}
while galaxy_busy; do sleep 120; done
say '"event": "galaxy_idle"'

export FRONTIERAM_GALAXY_DATA=public FRONTIERAM_GALAXY_V4_TIMING=600
id=$(./bench --store $PUB suite --task galaxy-gravity-v4 --reference --cases '["fb-4194304-disc"]' --shards 1 \
     | python3 -c "import json,sys;print(json.loads(sys.stdin.readlines()[-1])['id'])") || { say '"event": "timing_queue_failed"'; exit 1; }
./bench --store $PUB grade-worker --task galaxy-gravity-v4 --workers 1 --drain >> $LOG 2>&1
unset FRONTIERAM_GALAXY_V4_TIMING
python3 - "$PUB" "$id" <<'PY' >> $LOG || { say '"event": "timing_failed"'; exit 1; }
import json, math, sys
from datetime import datetime, timezone
store, suite = sys.argv[1:]
job = json.load(open(f'{store}/suites/{suite}.json'))['shards'][0]['job']
parts = json.load(open(f'{store}/results/{job}/grade/metrics.json'))['parts']
case = {w: next(c for c in parts[w]['cases'] if c['case'] == 'fb-4194304-disc') for w in ('strong', 'weak')}
if any(parts[w]['status'] != 'ok' or not case[w].get('cpu_s') for w in case):
    sys.exit(1)
strong_us = round(case['strong']['cpu_s']['max'] * 1e6)
limit = 100000 * math.ceil(5 * strong_us / 400000) / 1e6
path = 'tasks/galaxy-gravity/versions/v4/data/public/local-limits.json'
data = json.load(open(path))
data['limits']['fb-4194304-disc'] = limit
data['cases'] = [c for c in data['cases'] if c['case'] != 'fb-4194304-disc'] + [{
    'case': 'fb-4194304-disc', 'n': 4194304, 'family': 'disc', 'strong_max_cpu_s': case['strong']['cpu_s']['max'],
    'limit_s': limit, 'weak_max_cpu_s': case['weak']['cpu_s']['max'],
    'weak_fraction': case['weak']['cpu_s']['max'] / limit, 'timing_suite': suite,
    'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}]
data.pop('provisional', None)
open(path, 'w').write(json.dumps(data, indent=1) + '\n')
print(json.dumps({'event': 'limit', 'case': 'fb-4194304-disc', 'strong_max_cpu_s': case['strong']['cpu_s']['max'],
                  'weak_max_cpu_s': case['weak']['cpu_s']['max'], 'limit_s': limit}))
PY
.venv-ops/bin/python scripts/seal/prepare_galaxy_v4.py --feedback-and-workspace >> $LOG 2>&1 || { say '"event": "refresh_failed"'; exit 1; }

./bench public-baseline-v63 --task galaxy-gravity-v4 --replace >> $LOG 2>&1
python3 -c "
import json, sys
b = json.load(open('$PUB/public-baselines/galaxy-gravity-v4.json'))
sys.exit(0 if all(len(b[w]) == 16 and all(r['status'] == 'passed' for r in b[w]) for w in ('weak', 'strong')) else 1)" \
    || { say '"event": "feedback_baseline_failed"'; exit 1; }
say '"event": "feedback_baseline_16"'

sub=$(python3 -c "import json;print(json.load(open('tasks/galaxy-gravity/solutions/v6.3/v4/gpt-6.2-sol-single-1/score.json'))['candidate_suite'])")
sub=$(python3 -c "import json;print(json.load(open('var/stores/store-v63/suites/$sub.json'))['submission'])")
rm -rf $CHECK && mkdir -p $CHECK/src && tar -xf var/stores/store-v63/submissions/$sub/source.tar -C $CHECK/src
start=$(date +%s)
./bench public-eval-v63 --task galaxy-gravity-v4 --source $CHECK/src --out $CHECK/out --cap 900 --label "4.19M feedback check" >> $LOG 2>&1
python3 - "$CHECK/out/report.json" "$(( $(date +%s) - start ))" <<'PY' >> $LOG || { say '"event": "feedback_check_failed"'; exit 1; }
import json, sys
r = json.load(open(sys.argv[1]))
print(json.dumps({'event': 'feedback_check', 'status': r.get('status'), 'score': r.get('score'), 'total': r.get('total'),
                  'seconds': int(sys.argv[2])}))
sys.exit(0 if r.get('status') == 'ok' and (r.get('total') or {}).get('cases') == 16 else 1)
PY
say '"event": "chain_done"'
