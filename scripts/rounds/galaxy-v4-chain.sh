#!/usr/bin/env bash
# galaxy-gravity-v4 bring-up after the feedback pool (tasks/galaxy-gravity/analysis/B4_PRECALIBRATION.md), unattended:
# feedback-pool limit calibration, reseal (formal), public feedback baseline, then gpt-6.1-sol and claude-opus-5-5
# single + iterative attempts and their grade loops (the sol loop also anchors v4 under the frozen limits).
# Log: var/stores/store/reruns/galaxy-v4-chain.log
set -euo pipefail
cd "$(dirname "$0")/../.."
STORE=var/stores/store-v63
export FRONTIERAM_RUNTIME=$PWD/.runtime-recal
say() { echo "{\"time\": \"$(date +%FT%T)\", $1}"; }

while pgrep -f make_galaxy_v4_feedback.py > /dev/null; do sleep 30; done
[ -f tasks/galaxy-gravity/versions/v4/data/public/manifest.json ] || { say '"event": "feedback_pool_missing"'; exit 1; }
.venv-ops/bin/python scripts/seal/prepare_galaxy_v4.py --force > /dev/null
say '"event": "feedback_pool_sealed"'

suite=$(FRONTIERAM_GALAXY_DATA=public FRONTIERAM_GALAXY_V4_TIMING=300 ./bench --store $STORE suite --task galaxy-gravity-v4 \
        --reference --shards 5 --isolation shared | tail -1 | python3 -c "import json,sys;print(json.load(sys.stdin)['id'])")
FRONTIERAM_GALAXY_DATA=public FRONTIERAM_GALAXY_V4_TIMING=300 ./bench --store $STORE grade-worker --task galaxy-gravity-v4 \
    --workers 5 --drain > /dev/null 2>&1
.venv-ops/bin/python scripts/analyze/galaxy-v4-calibrate.py --store $STORE --suite "$suite" --public
.venv-ops/bin/python scripts/seal/prepare_galaxy_v4.py --force
grep -q '"status": "formal"' tasks/galaxy-gravity/versions/v4/task.json || { say '"event": "not_formal"'; exit 1; }
say "\"event\": \"sealed_formal\", \"feedback_timing_suite\": \"$suite\""

./bench public-baseline-v63 --task galaxy-gravity-v4
say '"event": "public_baseline_done"'

for model in gpt-6.1-sol claude-opus-5-5; do
    V63_CONCURRENCY=2 V63_MODEL=$model V63_TASKS=galaxy-gravity-v4 setsid nohup scripts/rounds/v63-sol.sh \
        > var/stores/store/reruns/v63-$model-galaxy-v4-runner.log 2>&1 < /dev/null &
done
say '"event": "attempts_started"'
# v4 grading (7 grade slots for ~30 min per submission) waits until the remaining v6.3 cells are graded
for label in opus-iterative-conic-1 opus-iterative-galaxy-gravity-v3-1 opus-iterative-ihvp-sequence-v3-1; do
    until grep -q "\"event\": \"graded\", \"label\": \"$label\"" var/stores/store/reruns/v63-grade.log; do sleep 60; done
done
V63_TASKS=galaxy-gravity-v4 setsid nohup scripts/rounds/v63-grade.sh >> var/stores/store/reruns/v63-grade-v4.out 2>&1 < /dev/null &
V63_MODEL=claude-opus-5-5 V63_TASKS=galaxy-gravity-v4 setsid nohup scripts/rounds/v63-grade.sh \
    >> var/stores/store/reruns/v63-grade-opus-v4.out 2>&1 < /dev/null &
say '"event": "grading_started"'
