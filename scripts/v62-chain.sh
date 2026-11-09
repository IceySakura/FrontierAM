#!/bin/bash
# Protocol v6.2 pipeline once the exponential-v2 private panel suite and the conic public baseline are done:
#   1. exponential-v2 public baseline (panel on the public set, one repeat)
#   2. private grading of gpt-6.1-sol's exponential-v2 single-setting attempt (same shard/slot map as the panel)
#   3. iterative-setting smoke checks (scripted agent, both tasks); stop here if either fails
#   4. gpt-6.1-sol iterative attempts, one at a time (test evaluations need every grading slot)
#   5. private grading of the remaining attempts (exponential-v2 iterative, conic single and iterative)
# Log: store/reruns/v62-chain.log; scores in results/v62-<label>/.
set -uo pipefail
cd "$(dirname "$0")/.."
R=$PWD
LOG=store/reruns/v62-chain.log
SLOTS=grade-06,grade-14,grade-22,grade-30
KEY=/root/.frontieram/aihub.key
PANEL=suite-exponential-v2-20261109T032227Z-c09381
CONIC_RT=$R/.runtime-conic-solver1105
EXP_RT=$R/.runtime-recal
P_B=0.7892307692307692
log() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> "$LOG"; }
last_id() { python3 -c "import json,sys;print(json.loads(sys.stdin.readlines()[-1])['id'])"; }

# RESUME=smoke skips the steps before the smoke checks (public baselines, the single-setting exponential-v2 grading).
RESUME=${RESUME:-}
if [ -z "$RESUME" ]; then
until grep -qE '"event": "(scored|score_failed|partition_mismatch|queue_failed)"' store/reruns/exp2-panel-suite.log; do sleep 60; done
while pgrep -f "public-baseline --task conic" >/dev/null; do sleep 60; done
log '"event": "start"'

FRONTIERAM_RUNTIME=$EXP_RT FRONTIERAM_EXP2_SEAL=$R/.runtime-exp2-public ./bench --store store-public-exp2 \
    public-baseline --task exponential-v2 > store/reruns/v62-public-baseline-exp2.log 2>&1 \
    || { log '"event": "exp2_public_baseline_failed"'; exit 1; }
log '"event": "public_baselines_done"'
fi

grade_exp2() {
    local submission=$1 label=$2 suite
    suite=$(FRONTIERAM_RUNTIME=$EXP_RT ./bench --store store-v62 suite --task exponential-v2 --submission "$submission" \
            --shards 4 --slots $SLOTS --isolation shared | last_id) || { log "\"event\": \"queue_failed\", \"label\": \"$label\""; return 1; }
    FRONTIERAM_RUNTIME=$EXP_RT ./bench --store store-v62 grade-worker --task exponential-v2 --workers 4 --drain >> store/reruns/v62-worker.log 2>&1
    .venv-ops/bin/python scripts/exp2-suite-score.py --store store-v62 --panel-store store --panel $PANEL \
        --candidate "$suite" --out "results/v62-$label" >> store/reruns/v62-score.log 2>&1
    log "\"event\": \"graded\", \"label\": \"$label\", \"suite\": \"$suite\", \"submission\": \"$submission\""
}

grade_conic() {
    local submission=$1 label=$2 suite
    suite=$(FRONTIERAM_RUNTIME=$CONIC_RT ./bench --store store-v62 suite --task conic --submission "$submission" \
            --shards 4 --slots $SLOTS --isolation shared | last_id) || { log "\"event\": \"queue_failed\", \"label\": \"$label\""; return 1; }
    FRONTIERAM_RUNTIME=$CONIC_RT ./bench --store store-v62 grade-worker --task conic --workers 4 --drain >> store/reruns/v62-worker.log 2>&1
    mkdir -p "results/v62-$label"
    FRONTIERAM_RUNTIME=$CONIC_RT ./bench --store store-v62 suite-report --suite "$suite" > /dev/null 2>&1
    python3 - "$suite" "$label" "$P_B" <<'PY'
import json, sys
suite, label, p_b = sys.argv[1], sys.argv[2], float(sys.argv[3])
report = json.load(open(f'store-v62/suites/{suite}.report.json'))
p = report['success_rate']
score = None if p is None or not report['complete'] else (
    0.0 if p <= 0.9 * p_b else 50 * (p - 0.9 * p_b) / (0.1 * p_b) if p <= p_b else 50 + 50 * (p - p_b) / (1 - p_b))
json.dump({'suite': suite, 'complete': report['complete'], 'successes': report['successes'],
           'cases': report['cases_graded'], 'p': p, 'p_B': p_b, 'score': score, 'groups': report['groups']},
          open(f'results/v62-{label}/score.json', 'w'), indent=1)
PY
    log "\"event\": \"graded\", \"label\": \"$label\", \"suite\": \"$suite\", \"submission\": \"$submission\""
}

develop_iterative() {
    local task=$1 runtime=$2 label=$3
    FRONTIERAM_RUNTIME=$runtime ./bench --store store-v62 develop --task "$task" --model gpt-6.1-sol --protocol v6.2 \
        --setting iterative --no-grade --key-file $KEY > "store/reruns/sol-v62-iterative-$label.log" 2>&1
    python3 -c "import json;print([json.loads(l) for l in open('store/reruns/sol-v62-iterative-$label.log') if 'develop_finished' in l][-1]['submission'])"
}

[ -z "$RESUME" ] && grade_exp2 exponential-v2-20261109T064531Z-785a70 sol-single-exp2
log "\"event\": \"resume\", \"from\": \"${RESUME:-start}\""

for task in conic exponential-v2; do
    runtime=$EXP_RT; [ $task = conic ] && runtime=$CONIC_RT
    FRONTIERAM_RUNTIME=$runtime PYTHONPATH=$R nexus/.venv/bin/python scripts/v62-smoke.py $task iterative 900 \
        "store/reruns/v62-smoke/$task-iterative-$(date +%H%M%S)" > "store/reruns/v62-smoke-$task-iterative.log" 2>&1 \
        || { log "\"event\": \"smoke_failed\", \"task\": \"$task\""; exit 1; }
    log "\"event\": \"smoke_passed\", \"task\": \"$task\""
done

exp_iter=$(develop_iterative exponential-v2 "$EXP_RT" exp2) || { log '"event": "develop_failed", "task": "exponential-v2"'; exit 1; }
log "\"event\": \"attempt_frozen\", \"task\": \"exponential-v2\", \"submission\": \"$exp_iter\""
conic_iter=$(develop_iterative conic "$CONIC_RT" conic) || { log '"event": "develop_failed", "task": "conic"'; exit 1; }
log "\"event\": \"attempt_frozen\", \"task\": \"conic\", \"submission\": \"$conic_iter\""

grade_exp2 "$exp_iter" sol-iterative-exp2
grade_conic conic-20261109T065544Z-ff7b81 sol-single-conic
grade_conic "$conic_iter" sol-iterative-conic
log '"event": "done"'
