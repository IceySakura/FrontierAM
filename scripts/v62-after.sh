#!/usr/bin/env bash
# After scripts/v62-final.sh exits: regrade the opus polarization attempt once (its first grading at 11-10 03:53 failed
# only on the first scene, 1.003 s against the 1.000 s limit, while the other 31 scenes took 0.56-0.58 s; same rule as
# the 11-06 sol regrade, see results/NEW_TASKS.md).
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=store/reruns/v62-final.log
log() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> "$LOG"; }
while pgrep -f scripts/v62-final.sh > /dev/null; do sleep 60; done
FRONTIERAM_RUNTIME=$PWD/.runtime-recal ./bench --store store grade --submission polarization-demosaicking-20261109T124343Z-553533 \
    --task polarization-demosaicking >> store/reruns/v62-worker.log 2>&1
log '"event": "regraded", "label": "opus-single-polarization", "submission": "polarization-demosaicking-20261109T124343Z-553533"'
