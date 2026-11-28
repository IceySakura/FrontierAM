#!/usr/bin/env bash
# GLM-5.3 on AI Hub, single setting only (user request 2026-11-28, while GLM-5.4 answered AI Hub 1011 to every
# request): Claude Code over /standard/v1/messages (ATR's glm-5.3 entry; the protocol tool lists only Chat
# Completions, but Messages answered with signed thinking and tool use). Runs the single attempt of the 10 board tasks
# (at most MAX at a time, the tasks with the longest grading first) and grades and publishes each cell.
# Log: var/stores/store/reruns/glm53-single.log
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=var/stores/store/reruns/glm53-single.log
QUEUE=var/stores/store/reruns/glm53-single-queue.txt
MAX=${MAX:-5}
say() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> $LOG; }

for task in galaxy-gravity-v4 conic species-tree-v3 exponential-v3 trajectory-covariance-v2 blackbox-cubature-v2 \
            ihvp-sequence-v4 polarization-demosaicking-v3 continuum-estimation fluorescence-deconvolution; do
    echo "glm-5.3 single $task 1"
done > $QUEUE
scripts/rounds/v63-lane.sh $QUEUE "$MAX" >> $LOG 2>&1 &
sleep 30
while read -r model _ task n; do
    V63_MODEL=$model V63_ATTEMPT=$n V63_TASKS=$task scripts/rounds/v63-grade.sh >> var/stores/store/reruns/glm53-single-grade.log 2>&1 &
done < $QUEUE
say '"event": "attempts_started"'
wait
say '"event": "chain_done"'
