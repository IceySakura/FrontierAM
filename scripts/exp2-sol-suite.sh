#!/bin/bash
# exponential-v2 on the sealed private split: the frozen panel and gpt-6.1-sol's v6.1 attempt, each as a 4-shard suite
# with the same case partition and shard-to-slot map (each case measured for the panel and the candidate on the same
# slot, the four slots in separate L3 domains running in parallel as for the conic suites), then scored by
# scripts/exp2-suite-score.py. Waits for the conic suite (it needs every grading slot). Log: store/reruns/exp2-sol-suite.log
set -uo pipefail
cd "$(dirname "$0")/.."
export FRONTIERAM_RUNTIME=${FRONTIERAM_RUNTIME:-/data/workspace/FrontierAM/.runtime-recal}
LOG=store/reruns/exp2-sol-suite.log
log() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> "$LOG"; }
until grep -q private_sealed store/reruns/exp2-freeze.log 2>/dev/null; do
    grep -qE 'private_failed|freeze_failed' store/reruns/exp2-freeze.log 2>/dev/null && { log '"event": "no_private_seal"'; exit 1; }
    sleep 60
done
until grep -q suite_done store/reruns/conic2-sol-suite.log 2>/dev/null; do sleep 60; done
submission=$(python3 -c "import json;print([json.loads(l) for l in open('store/reruns/sol-v61-exponential-v2.log') if 'develop_finished' in l][-1]['submission'])")
SLOTS=grade-06,grade-14,grade-22,grade-30
queue() { ./bench --store store suite --task exponential-v2 "$@" --shards 4 --slots $SLOTS --isolation shared |
          python3 -c "import json,sys;print(json.loads(sys.stdin.readlines()[-1])['id'])"; }
panel=$(queue --reference)
candidate=$(queue --submission "$submission")
log "\"event\": \"suites_queued\", \"panel\": \"$panel\", \"candidate\": \"$candidate\", \"submission\": \"$submission\""
./bench --store store grade-worker --task exponential-v2 --workers 4 --drain >> store/reruns/exp2-worker.log 2>&1
out=results/exp2-sol-$(date +%Y%m%d)
.venv-ops/bin/python scripts/exp2-suite-score.py --store store --panel "$panel" --candidate "$candidate" --out "$out" >> "$LOG" 2>&1
log "\"event\": \"scored\", \"out\": \"$out\""
