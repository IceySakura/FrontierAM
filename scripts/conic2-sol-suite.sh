#!/bin/bash
# conic author revision (conic-solver-20261105): grade gpt-6.1-sol's v6.1 attempt on the full 650-case suite with the
# locally recalibrated limits, in its own store so that no worker of the earlier runtimes can take its shards.
# Waits for the attempt to freeze and for the exponential-v2 private references (they load every core) to finish.
# Log: store/reruns/conic2-sol-suite.log
set -uo pipefail
cd "$(dirname "$0")/.."
export FRONTIERAM_RUNTIME=/data/workspace/FrontierAM/.runtime-conic-solver1105
LOG=store/reruns/conic2-sol-suite.log
log() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> "$LOG"; }
until grep -q develop_finished store/reruns/sol-v61-conic-solver1105.log; do sleep 60; done
submission=$(python3 -c "import json;print([json.loads(l) for l in open('store/reruns/sol-v61-conic-solver1105.log') if 'develop_finished' in l][-1]['submission'])")
log "\"event\": \"attempt_frozen\", \"submission\": \"$submission\""
until grep -qE 'private_sealed|private_failed|freeze_failed' store/reruns/exp2-freeze.log 2>/dev/null; do sleep 60; done
log '"event": "machine_free_of_reference_computation"'
mkdir -p store-conic2/submissions
[ -d "store-conic2/submissions/$submission" ] || cp -a "store/submissions/$submission" "store-conic2/submissions/$submission"
suite=$(./bench --store store-conic2 suite --task conic --submission "$submission" --shards 4 \
        --slots grade-06,grade-14,grade-22,grade-30 | python3 -c "import json,sys;print(json.loads(sys.stdin.readlines()[-1])['id'])")
log "\"event\": \"suite_queued\", \"suite\": \"$suite\""
./bench --store store-conic2 grade-worker --task conic --workers 4 --drain >> store/reruns/conic2-worker.log 2>&1
./bench --store store-conic2 suite-report --suite "$suite" > "store/reruns/conic2-sol-report.json" 2>&1
log "\"event\": \"suite_done\", \"suite\": \"$suite\", \"report\": \"store/reruns/conic2-sol-report.json\""
