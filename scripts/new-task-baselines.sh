#!/bin/bash
# Unattended: once the grading daemons hold the new images and no other baseline runs, measure each new task's
# reference baseline on a quiet machine (one at a time), run disk maintenance while the machine is still idle,
# then hand the grading slots back to the held conic shards.
# Usage: scripts/new-task-baselines.sh TASK [TASK ...]
set -uo pipefail
cd "$(dirname "$0")/.."
export FRONTIERAM_RUNTIME=${FRONTIERAM_RUNTIME:-.runtime-suite}
log() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> store/reruns/new-task-baselines.log; }
while pgrep -f load-grade-images.sh > /dev/null || pgrep -f "arena.cli baseline" > /dev/null; do sleep 30; done
log '"event": "machine_ready"'
for task in "$@"; do
    log "\"event\": \"baseline_start\", \"task\": \"$task\""
    ./bench baseline --task "$task" > "store/reruns/baseline-$task.log" 2>&1
    log "\"event\": \"baseline_done\", \"task\": \"$task\", \"exit\": $?"
done
scripts/data-maintenance.sh
log "\"event\": \"data_maintenance_done\", \"exit\": $?"
scripts/exp2-public.sh
log "\"event\": \"exp2_public_done\", \"exit\": $?"
python3 scripts/release-when-free.py store --defer store-recal >> store/reruns/release-main.log 2>&1 < /dev/null &
log '"event": "conic_release_resumed"'
