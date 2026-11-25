#!/usr/bin/env bash
# Regrade one frozen v6.3 submission of a timing-sensitive task (polarization-demosaicking: 1 s per scene;
# exponential-v3: per-case CPU caps, which development load inflates) once no development slot is leased (dev_idle
# isolation; gradings still run side by side), after its first grading met development load. The first score.json is
# kept as score.first.json; the new one is written in the format of scripts/rounds/v63-grade.sh, then the round is
# republished if the model is on the board (scripts/rounds/v63-publish.py MODELS).
# Usage: regrade-dev-idle.sh TASK SUBMISSION SOLUTION_DIR   (log: var/stores/store/reruns/regrade-dev-idle.log)
set -uo pipefail
cd "$(dirname "$0")/../.."
task=$1 sub=$2 out=$3
STORE=var/stores/store-v63
LOG=var/stores/store/reruns/regrade-dev-idle.log
PANEL=suite-exponential-v2-20261109T032227Z-c09381   # as scripts/rounds/v63-grade.sh
export FRONTIERAM_RUNTIME=$PWD/.runtime-recal
started=$(date -u +%FT%T)
echo "{\"time\": \"$(date +%FT%T)\", \"event\": \"regrade_waiting\", \"task\": \"$task\", \"submission\": \"$sub\"}" >> $LOG
if [ "$task" = exponential-v3 ]; then
    # The suite's shards are queued only once development is idle: queued dev_idle shards would hold up the
    # exponential gradings of other cells, which share the task's queue.
    dev_busy() { python3 -c "import sys;sys.path.insert(0,'scripts/ops');import slot_status as s;sys.exit(0 if any(c['role']=='dev' and c.get('lease') for c in s.snapshot()['cores']) else 1)"; }
    while dev_busy; do sleep 60; done
    id=$(./bench --store $STORE suite --task "$task" --submission "$sub" --shards 7 --isolation dev_idle --cap-s 3600 \
         | python3 -c "import json,sys;print(json.loads(sys.stdin.readlines()[-1])['id'])") || exit 1
    ./bench --store $STORE grade-worker --task "$task" --workers 7 --drain >> var/stores/store/reruns/v63-worker.log 2>&1
    # another worker of the same task may still be grading some shards; a shard in no queue and not done is lost
    while state=$(python3 -c "
import json,os
r=json.load(open('$STORE/suites/$id.json')); q='$STORE/jobs/'
s=[next((k for k in ('done','pending','running','held') if os.path.exists(q+k+'/'+x['job']+'.json')),'lost') for x in r['shards']]
print('lost' if 'lost' in s else 'done' if all(k=='done' for k in s) else 'wait')"); [ "$state" = wait ]; do
        sleep 30
    done
    [ "$state" = done ] || { echo "{\"event\": \"regrade_shard_lost\", \"suite\": \"$id\"}" >> $LOG; exit 1; }
    [ -f "$out/score.first.json" ] || mv "$out/score.json" "$out/score.first.json"
    .venv-ops/bin/python scripts/grade/exp2-suite-score.py --task "$task" --store $STORE --panel-store var/stores/store \
        --panel $PANEL --candidate "$id" --out "$out" >> var/stores/store/reruns/v63-score.log 2>&1
    python3 -c "import json,sys;d=json.load(open(sys.argv[1]+'/score.json'));print(json.dumps({'event':'regraded','submission':sys.argv[2],'suite':sys.argv[3],'status':d.get('status'),'relative_time':d.get('relative_time'),'score':d.get('score')}))" \
        "$out" "$sub" "$id" >> $LOG
else
    ./bench --store $STORE grade --submission "$sub" --task "$task" --isolation dev_idle \
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
fi
model=$(basename "$out" | sed -E 's/-(single|iterative)-[0-9]+$//')
grep -q "('$model', '" scripts/rounds/v63-publish.py && scripts/rounds/publish-now.sh "regrade $task $sub"
