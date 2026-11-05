#!/bin/bash
# Unattended: once the grading daemons hold the new images, measure each new task's reference baseline on a quiet
# machine (one at a time), then hand the grading slots back to the held conic shards.
# Usage: scripts/new-task-baselines.sh TASK [TASK ...]
set -uo pipefail
cd "$(dirname "$0")/.."
export FRONTIERAM_RUNTIME=${FRONTIERAM_RUNTIME:-.runtime-suite}
log() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> store/reruns/new-task-baselines.log; }
while pgrep -f load-grade-images.sh > /dev/null; do sleep 30; done
log '"event": "images_loaded"'
for task in "$@"; do
    log "\"event\": \"baseline_start\", \"task\": \"$task\""
    ./bench baseline --task "$task" > "store/reruns/baseline-$task.log" 2>&1
    log "\"event\": \"baseline_done\", \"task\": \"$task\", \"exit\": $?"
done
python3 scripts/release-when-free.py store --defer store-recal >> store/reruns/release-main.log 2>&1 < /dev/null &
log '"event": "conic_release_resumed"'
