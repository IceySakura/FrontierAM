#!/usr/bin/env bash
# Regrade one frozen v6.3 submission of a timing-sensitive task (polarization-demosaicking: 1 s per scene) once no
# development slot is leased (dev_idle isolation; gradings still run side by side), after its first grading met
# development load. The first score.json is kept as score.first.json; the new one is written in the format of
# scripts/rounds/v63-grade.sh, then the round is republished.
# Usage: regrade-dev-idle.sh TASK SUBMISSION SOLUTION_DIR   (log: var/stores/store/reruns/regrade-dev-idle.log)
set -uo pipefail
cd "$(dirname "$0")/../.."
task=$1 sub=$2 out=$3
STORE=var/stores/store-v63
LOG=var/stores/store/reruns/regrade-dev-idle.log
started=$(date -u +%FT%T)
echo "{\"time\": \"$(date +%FT%T)\", \"event\": \"regrade_waiting\", \"task\": \"$task\", \"submission\": \"$sub\"}" >> $LOG
FRONTIERAM_RUNTIME=$PWD/.runtime-recal ./bench --store $STORE grade --submission "$sub" --task "$task" --isolation dev_idle \
    >> var/stores/store/reruns/v63-worker.log 2>&1
python3 - "$STORE" "$sub" "$out" "$started" >> $LOG <<'PY'
import json, sys
from pathlib import Path
store, sub, out, started = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
done = [json.loads(p.read_text()) for p in (store / 'jobs/done').glob('*.json')]
rows = sorted((d for d in done if d.get('submission') == sub and d.get('kind') == 'candidate'
               and (d.get('finished_utc') or '')[:19] > started), key=lambda d: d['finished_utc'])
if not rows:
    print(json.dumps({'event': 'regrade_missing', 'submission': sub}))
    sys.exit(1)
record = rows[-1]
metrics = store / 'results' / record['id'] / 'grade/metrics.json'
if metrics.exists():
    record['candidate'] = json.loads(metrics.read_text())['parts'].get('candidate')
if (out / 'score.json').exists() and not (out / 'score.first.json').exists():
    (out / 'score.json').rename(out / 'score.first.json')
json.dump(record, open(out / 'score.json', 'w'), indent=1)
candidate = record.get('candidate') or {}
print(json.dumps({'event': 'regraded', 'submission': sub, 'job': record['id'], 'status': candidate.get('status'),
                  'raw_error': candidate.get('raw_error'), 'max_wall_s': candidate.get('max_wall_s')}))
PY
scripts/rounds/publish-now.sh "regrade $task $sub"
