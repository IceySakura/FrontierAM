#!/bin/bash
# exponential-v2: once the public calibrations finish, select under calibration_v2/RULE.md, freeze the panel, then
# (only then) generate the private split with its references and seal the task. Log: var/stores/store/reruns/exp2-freeze.log
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
C=tasks/exponential/references/r1/calibration
LOG=var/stores/store/reruns/exp2-freeze.log
log() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> "$LOG"; }
while docker ps --no-trunc --format '{{.Command}}' | grep -q calibrate_v2; do sleep 60; done
log '"event": "calibrations_finished"'
docker run --rm --network none -v $ROOT:$ROOT -w $ROOT --entrypoint python3 frontieram-exponential:port-v1 \
    $C/calibrate_v2.py select --manifest $ROOT/tasks/exponential/versions/v2/dataset/cases/public/manifest.json \
    --out $ROOT/$C/v2 >> "$LOG" 2>&1
.venv-ops/bin/python scripts/seal/exp2-freeze.py >> "$LOG" 2>&1 || { log '"event": "freeze_failed"'; exit 1; }
log "\"event\": \"frozen\", \"members\": $(python3 -c "import json;print(json.dumps(list(json.load(open('$C/FROZEN.json'))['members'])))")"
TRUTH_CPUS=${TRUTH_CPUS:-6-31} scripts/references/exp2-private.sh >> "$LOG" 2>&1 || { log '"event": "private_failed"'; exit 1; }
log '"event": "private_sealed"'
