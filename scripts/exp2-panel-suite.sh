#!/bin/bash
# exponential-v2 panel suite on the sealed private split, sharded exactly like an existing candidate suite (same case
# partition and shard-to-slot map), then scored against it by scripts/exp2-suite-score.py.
# Usage: exp2-panel-suite.sh CANDIDATE_SUITE   Log: var/stores/store/reruns/exp2-panel-suite.log
set -uo pipefail
cd "$(dirname "$0")/.."
export FRONTIERAM_RUNTIME=${FRONTIERAM_RUNTIME:-/data/workspace/FrontierAM/.runtime-recal}
LOG=var/stores/store/reruns/exp2-panel-suite.log
log() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> "$LOG"; }
candidate=$1
panel=$(./bench --store var/stores/store suite --task exponential-v2 --reference --shards 4 --slots grade-06,grade-14,grade-22,grade-30 \
        --isolation shared | python3 -c "import json,sys;print(json.loads(sys.stdin.readlines()[-1])['id'])") || { log '"event": "queue_failed"'; exit 1; }
log "\"event\": \"panel_queued\", \"panel\": \"$panel\", \"candidate\": \"$candidate\""
./bench --store var/stores/store grade-worker --task exponential-v2 --workers 4 --drain >> var/stores/store/reruns/exp2-worker.log 2>&1
log "\"event\": \"panel_graded\", \"panel\": \"$panel\""
python3 - "$panel" "$candidate" <<'PY' || { log "\"event\": \"partition_mismatch\", \"panel\": \"$panel\""; exit 1; }
import json, sys
def key(suite):
    shards = json.load(open(f'var/stores/store/suites/{suite}.json'))['shards']
    return [(json.load(open(f"var/stores/store/results/{s['job']}/result.json"))['slot'],
             sorted(json.load(open(f"var/stores/store/results/{s['job']}/spec.json"))['case_ids'])) for s in shards]
sys.exit(0 if key(sys.argv[1]) == key(sys.argv[2]) else 1)
PY
out=results/exponential/exp2-sol-$(date +%Y%m%d)
.venv-ops/bin/python scripts/exp2-suite-score.py --store var/stores/store --panel "$panel" --candidate "$candidate" --out "$out" >> "$LOG" 2>&1 \
    && log "\"event\": \"scored\", \"out\": \"$out\"" || log '"event": "score_failed"'
