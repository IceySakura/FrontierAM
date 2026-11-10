#!/bin/bash
# After scripts/v62-chain.sh finishes: grade gpt-6.1-sol's exponential-v2 single-setting rerun (the attempt with the
# corrected timing note, results/PROTOCOL_V62.md) on the private set, same shard/slot map as the panel.
set -uo pipefail
cd "$(dirname "$0")/.."
R=$PWD
LOG=var/stores/store/reruns/v62-chain.log
log() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> "$LOG"; }
until grep -qE '"event": "(done|develop_failed|smoke_failed|queue_failed)"' "$LOG"; do sleep 60; done
submission=exponential-v2-20261109T090735Z-9ccd95
suite=$(FRONTIERAM_RUNTIME=$R/.runtime-recal ./bench --store var/stores/store-v62 suite --task exponential-v2 --submission $submission \
        --shards 4 --slots grade-06,grade-14,grade-22,grade-30 --isolation shared | python3 -c "import json,sys;print(json.loads(sys.stdin.readlines()[-1])['id'])")
FRONTIERAM_RUNTIME=$R/.runtime-recal ./bench --store var/stores/store-v62 grade-worker --task exponential-v2 --workers 4 --drain >> var/stores/store/reruns/v62-worker.log 2>&1
.venv-ops/bin/python scripts/exp2-suite-score.py --store var/stores/store-v62 --panel-store store --panel suite-exponential-v2-20261109T032227Z-c09381 \
    --candidate "$suite" --out results/runs/v62-sol-single-exp2-rerun >> var/stores/store/reruns/v62-score.log 2>&1
log "\"event\": \"graded\", \"label\": \"sol-single-exp2-rerun\", \"suite\": \"$suite\", \"submission\": \"$submission\""
