#!/usr/bin/env bash
# Reruns after the feedback fixes of species-tree-v3, exponential-v3 and galaxy-gravity-v4 (user decision 2026-11-27,
# results/OPEN_ITEMS.md): every iterative cell of the 8 board models (attempt numbers in v63-publish.py
# FEEDBACK_FIX_ATTEMPT), at most MAX attempts at a time, galaxy last; one grading loop per model and task
# (v63-grade.sh publishes each graded cell). Development starts only after the work that needs idle development slots:
# the scale pilot's polarization snapshots, then the CPU recheck of qwen3.8's exponential single; galaxy waits for its
# new feedback (galaxy-4m-feedback-chain.sh).
# Usage: feedback-fix-reruns.sh [MAX]   (log: var/stores/store/reruns/feedback-fix-reruns.log)
set -uo pipefail
cd "$(dirname "$0")/../.."
MAX=${1:-3}
LOG=var/stores/store/reruns/feedback-fix-reruns.log
GALAXY=var/stores/store/reruns/galaxy-4m-feedback.log
export FRONTIERAM_RUNTIME=$PWD/.runtime-recal
say() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> $LOG; }

until grep -q '^polarization done' var/raw/scale-curves/pilot-runner2.log 2>/dev/null; do sleep 120; done
say '"event": "pilot_polarization_done"'

out=tasks/exponential/solutions/v6.3/qwen3.8-max-0902-single-2
candidate=$(python3 -c "import json;print(json.load(open('$out/score.json'))['candidate_suite'])")
.venv-ops/bin/python scripts/grade/exp-cpu-recheck.py --store var/stores/store-v63 --candidate "$candidate" --out "$out" >> $LOG 2>&1
say '"event": "exp_recheck_done"'

until grep -q '"event": "chain_done"' $GALAXY 2>/dev/null; do
    grep -q 'failed"' $GALAXY 2>/dev/null && { say '"event": "galaxy_feedback_not_ready"'; exit 1; }
    sleep 120
done
say '"event": "galaxy_feedback_ready"'

QUEUE=var/stores/store/reruns/feedback-fix-queue.txt
python3 - > $QUEUE <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location('publish', 'scripts/rounds/v63-publish.py')
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)
for task in ('species-tree-v3', 'exponential-v3', 'galaxy-gravity-v4'):
    for model, n in publish.FEEDBACK_FIX_ATTEMPT[task].items():
        print(model, 'iterative', task, n)
PY
scripts/rounds/v63-lane.sh $QUEUE "$MAX" >> $LOG 2>&1 &
sleep 30
while read -r model _ task n; do
    V63_MODEL=$model V63_ATTEMPT=$n V63_TASKS=$task scripts/rounds/v63-grade.sh \
        >> var/stores/store/reruns/feedback-fix-grade.log 2>&1 &
done < $QUEUE
say '"event": "reruns_started"'
wait
say '"event": "chain_done"'
