#!/usr/bin/env bash
# GLM-5.4 on AI Hub, single setting only (user request 2026-11-28): wait until the model answers reliably on AI Hub
# (scripts/rounds/glm-health.py; on 11-28 it failed almost every request with 1011), then run the single attempt of the
# 10 board tasks (at most MAX at a time, the tasks with the longest grading first) and grade and publish each cell.
# Log: var/stores/store/reruns/glm-single.log
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=var/stores/store/reruns/glm-single.log
QUEUE=var/stores/store/reruns/glm-single-queue.txt
MAX=${MAX:-4}
say() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> $LOG; }

until line=$(python3 scripts/rounds/glm-health.py); do
    echo "$line" >> $LOG
    sleep 600
done
echo "$line" >> $LOG
say '"event": "aihub_healthy"'

for task in galaxy-gravity-v4 conic species-tree-v3 exponential-v3 trajectory-covariance-v2 blackbox-cubature-v2 \
            ihvp-sequence-v4 polarization-demosaicking-v3 continuum-estimation fluorescence-deconvolution; do
    echo "glm-5.4 single $task 1"
done > $QUEUE
scripts/rounds/v63-lane.sh $QUEUE "$MAX" >> $LOG 2>&1 &
sleep 30
while read -r model _ task n; do
    V63_MODEL=$model V63_ATTEMPT=$n V63_TASKS=$task scripts/rounds/v63-grade.sh >> var/stores/store/reruns/glm-single-grade.log 2>&1 &
done < $QUEUE
say '"event": "attempts_started"'
wait
say '"event": "chain_done"'
