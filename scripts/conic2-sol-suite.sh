#!/bin/bash
# conic author revision (conic-solver-20261105): grade gpt-6.1-sol's v6.1 attempt on the full 650-case suite with the
# locally recalibrated limits, in its own store so that no worker of the earlier runtimes can take its shards.
# Waits for the attempt to freeze and for the exponential-v2 private references (they load every core) to finish.
# Log: var/stores/store/reruns/conic2-sol-suite.log
set -uo pipefail
cd "$(dirname "$0")/.."
export FRONTIERAM_RUNTIME=/data/workspace/FrontierAM/.runtime-conic-solver1105
LOG=var/stores/store/reruns/conic2-sol-suite.log
log() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> "$LOG"; }
until grep -q develop_finished var/stores/store/reruns/sol-v61-conic-solver1105.log; do sleep 60; done
submission=$(python3 -c "import json;print([json.loads(l) for l in open('var/stores/store/reruns/sol-v61-conic-solver1105.log') if 'develop_finished' in l][-1]['submission'])")
log "\"event\": \"attempt_frozen\", \"submission\": \"$submission\""
until grep -qE 'private_sealed|private_failed|freeze_failed' var/stores/store/reruns/exp2-freeze.log 2>/dev/null; do sleep 60; done
log '"event": "machine_free_of_reference_computation"'
mkdir -p var/stores/store-conic2/submissions
[ -d "var/stores/store-conic2/submissions/$submission" ] || cp -a "var/stores/store/submissions/$submission" "var/stores/store-conic2/submissions/$submission"
suite=$(./bench --store var/stores/store-conic2 suite --task conic --submission "$submission" --shards 4 \
        --slots grade-06,grade-14,grade-22,grade-30 | python3 -c "import json,sys;print(json.loads(sys.stdin.readlines()[-1])['id'])")
log "\"event\": \"suite_queued\", \"suite\": \"$suite\""
./bench --store var/stores/store-conic2 grade-worker --task conic --workers 4 --drain >> var/stores/store/reruns/conic2-worker.log 2>&1
./bench --store var/stores/store-conic2 suite-report --suite "$suite" > "var/stores/store/reruns/conic2-sol-report.json" 2>&1
log "\"event\": \"suite_done\", \"suite\": \"$suite\", \"report\": \"var/stores/store/reruns/conic2-sol-report.json\""
