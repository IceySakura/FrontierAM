#!/usr/bin/env bash
# Regrade every v6.3 continuum-estimation cell on the 45 scored cases (2026-11-26: 15 of the 60 primary cases became
# the held-out public feedback, tasks/continuum-estimation/versions/v1/data/private/feedback-split.json), one cell at a
# time, against the active baseline re-measured on the same 45 cases. The 60-case score.json is kept as score.60.json;
# the new one has the format of scripts/rounds/v63-grade.sh. Cells already graded on 45 cases are skipped.
# Usage: regrade-continuum45.sh [SOLUTION_DIR ...]   (default: every cell; log: var/stores/store/reruns/regrade-continuum45.log)
set -uo pipefail
cd "$(dirname "$0")/../.."
STORE=var/stores/store-v63
LOG=var/stores/store/reruns/regrade-continuum45.log
export FRONTIERAM_RUNTIME=$PWD/.runtime-recal
cells=("$@")
[ ${#cells[@]} -gt 0 ] || cells=(tasks/continuum-estimation/solutions/v6.3/*/)
for out in "${cells[@]}"; do
    out=${out%/}
    sub=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('submission') or '')" "$out/score.json" 2>/dev/null)
    [ -n "$sub" ] || continue
    python3 -c "import json,sys;sys.exit(0 if (json.load(open(sys.argv[1])).get('candidate') or {}).get('cases') == 45 else 1)" \
        "$out/score.json" && continue
    started=$(date -u +%FT%T)
    ./bench --store $STORE grade --submission "$sub" --task continuum-estimation --isolation shared >> "$LOG" 2>&1
    python3 - "$STORE" "$sub" "$out" "$started" <<'PY' >> "$LOG"
import json, sys
from pathlib import Path
store, sub, out, started = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
done = [json.loads(p.read_text()) for p in (store / 'jobs/done').glob('*.json')]
rows = sorted((d for d in done if d.get('submission') == sub and d.get('kind') == 'candidate'
               and (d.get('finished_utc') or '')[:19] > started), key=lambda d: d['finished_utc'])
if not rows:
    print(json.dumps({'event': 'regrade_missing', 'cell': str(out), 'submission': sub}))
    sys.exit(1)
record = rows[-1]
metrics = store / 'results' / record['id'] / 'grade/metrics.json'
if metrics.exists():
    record['candidate'] = json.loads(metrics.read_text())['parts'].get('candidate')
if not (out / 'score.60.json').exists():
    (out / 'score.json').rename(out / 'score.60.json')
json.dump(record, open(out / 'score.json', 'w'), indent=1)
candidate = record.get('candidate') or {}
print(json.dumps({'event': 'regraded', 'cell': out.name, 'submission': sub, 'score': record.get('score'),
                  'cases': candidate.get('cases'), 'raw_error': candidate.get('raw_error')}))
PY
done
scripts/rounds/publish-now.sh "continuum-estimation on 45 scored cases"
