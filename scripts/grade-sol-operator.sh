#!/bin/bash
# Grade frozen operator-task submissions one at a time on an otherwise idle set of grading CPUs: accuracy-only
# calibration containers (exponential-v2 panel) are paused for the duration and resumed afterwards.
# Usage: scripts/grade-sol-operator.sh TASK=SUBMISSION [TASK=SUBMISSION ...]
set -uo pipefail
cd "$(dirname "$0")/.."
export FRONTIERAM_RUNTIME=${FRONTIERAM_RUNTIME:-/data/workspace/FrontierAM/.runtime-recal}
LOG=var/stores/store/reruns/grade-sol-operator.log
log() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> "$LOG"; }
paused=$(docker ps --filter ancestor=frontieram-exponential:port-v1 --no-trunc --format '{{.ID}} {{.Command}}' | grep calibrate_v2 | cut -c1-12)
for id in $paused; do docker pause "$id" > /dev/null && log "\"event\": \"paused\", \"container\": \"$id\""; done
for item in "$@"; do
    task=${item%%=*}; submission=${item#*=}
    log "\"event\": \"grade_start\", \"task\": \"$task\", \"submission\": \"$submission\""
    ./bench grade --task "$task" --submission "$submission" >> "var/stores/store/reruns/grade-sol-$task.log" 2>&1
    log "\"event\": \"grade_done\", \"task\": \"$task\", \"exit\": $?, \"last\": $(tail -1 "var/stores/store/reruns/grade-sol-$task.log")"
done
for id in $paused; do docker unpause "$id" > /dev/null && log "\"event\": \"unpaused\", \"container\": \"$id\""; done
