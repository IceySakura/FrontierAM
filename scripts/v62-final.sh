#!/bin/bash
# Protocol v6.2 under the final rules (15-minute test cap with partial results, submit --status, no other grading
# during iterative attempts), then the grading backlog. Log: store/reruns/v62-final.log
#   1. wait for the exponential-v2 iterative pilot (old prompt) to end
#   2. smoke checks: conic and exponential-v2 iterative, conic capped test (cap shortened for the check only)
#   3. gpt-6.1-sol iterative attempts under the final rules: conic, then exponential-v2
#   4. private grading: both exponential-v2 iterative attempts (pilot = record only)
#   5. species-tree-v2 reference anchors (sharded), once the task is sealed
#   6. private grading: conic single and iterative (long)
set -uo pipefail
cd "$(dirname "$0")/.."
R=$PWD
LOG=store/reruns/v62-final.log
SLOTS=grade-06,grade-14,grade-22,grade-30
KEY=/root/.frontieram/aihub.key
PANEL=suite-exponential-v2-20261109T032227Z-c09381
CONIC_RT=$R/.runtime-conic-solver1105
EXP_RT=$R/.runtime-recal
P_B=0.7892307692307692
log() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> "$LOG"; }
last_id() { python3 -c "import json,sys;print(json.loads(sys.stdin.readlines()[-1])['id'])"; }

until [ -f store-v62/dev-runs/dev-exponential-v2-20261109T104632Z-517812/develop_result.json ]; do sleep 60; done
pilot=$(python3 -c "import json;print([json.loads(l) for l in open('store/reruns/sol-v62-iterative-exp2.log') if 'develop_finished' in l][-1]['submission'])" 2>/dev/null)
log "\"event\": \"pilot_finished\", \"submission\": \"$pilot\""

smoke() {
    local task=$1 setting=$2 runtime=$3 cap=${4:-}
    FRONTIERAM_V62_TEST_CAP=$cap FRONTIERAM_RUNTIME=$runtime PYTHONPATH=$R nexus/.venv/bin/python scripts/v62-smoke.py \
        "$task" "$setting" 900 "store/reruns/v62-smoke/final-$task-$setting-$(date +%H%M%S)" \
        > "store/reruns/v62-smoke-final-$task-$setting.log" 2>&1 \
        || { log "\"event\": \"smoke_failed\", \"task\": \"$task\", \"setting\": \"$setting\""; exit 1; }
    log "\"event\": \"smoke_passed\", \"task\": \"$task\", \"setting\": \"$setting\""
}
smoke conic iterative "$CONIC_RT"
smoke exponential-v2 iterative "$EXP_RT"
smoke conic iterative-cap "$CONIC_RT" 60

develop_iterative() {
    local task=$1 runtime=$2 label=$3
    FRONTIERAM_RUNTIME=$runtime ./bench --store store-v62 develop --task "$task" --model gpt-6.1-sol --protocol v6.2 \
        --setting iterative --no-grade --key-file $KEY > "store/reruns/sol-v62-final-iterative-$label.log" 2>&1
    python3 -c "import json;print([json.loads(l) for l in open('store/reruns/sol-v62-final-iterative-$label.log') if 'develop_finished' in l][-1]['submission'])"
}
conic_iter=$(develop_iterative conic "$CONIC_RT" conic) || { log '"event": "develop_failed", "task": "conic"'; exit 1; }
log "\"event\": \"attempt_frozen\", \"task\": \"conic\", \"submission\": \"$conic_iter\""
exp_iter=$(develop_iterative exponential-v2 "$EXP_RT" exp2) || { log '"event": "develop_failed", "task": "exponential-v2"'; exit 1; }
log "\"event\": \"attempt_frozen\", \"task\": \"exponential-v2\", \"submission\": \"$exp_iter\""

grade_exp2() {
    local submission=$1 label=$2 suite
    suite=$(FRONTIERAM_RUNTIME=$EXP_RT ./bench --store store-v62 suite --task exponential-v2 --submission "$submission" \
            --shards 4 --slots $SLOTS --isolation shared | last_id) || { log "\"event\": \"queue_failed\", \"label\": \"$label\""; return 1; }
    FRONTIERAM_RUNTIME=$EXP_RT ./bench --store store-v62 grade-worker --task exponential-v2 --workers 4 --drain >> store/reruns/v62-worker.log 2>&1
    .venv-ops/bin/python scripts/exp2-suite-score.py --store store-v62 --panel-store store --panel $PANEL \
        --candidate "$suite" --out "results/v62-$label" >> store/reruns/v62-score.log 2>&1
    log "\"event\": \"graded\", \"label\": \"$label\", \"suite\": \"$suite\", \"submission\": \"$submission\""
}
grade_exp2 "$exp_iter" sol-iterative-exp2
[ -n "$pilot" ] && grade_exp2 "$pilot" sol-iterative-exp2-pilot

until [ -f .runtime-ops/species-tree-v2/lock.json ]; do sleep 60; done
species_ref=$(FRONTIERAM_RUNTIME=$EXP_RT ./bench --store store-species-v2 suite --task species-tree-v2 --reference --shards 4 \
              --slots $SLOTS --isolation shared | last_id) || { log '"event": "queue_failed", "label": "species-v2-reference"'; exit 1; }
FRONTIERAM_RUNTIME=$EXP_RT ./bench --store store-species-v2 grade-worker --task species-tree-v2 --workers 4 --drain >> store/reruns/v62-worker.log 2>&1
.venv-ops/bin/python scripts/species-v2-suite-score.py --store store-species-v2 --reference "$species_ref" \
    --out results/species-tree-v2/anchors.json >> store/reruns/v62-score.log 2>&1
log "\"event\": \"species_v2_anchors\", \"suite\": \"$species_ref\""

# galaxy-gravity-v2 reference pair (quiet, one job), then the new tasks' gpt-6.1-sol single attempts.
FRONTIERAM_RUNTIME=$EXP_RT ./bench --store store-galaxy-v2 baseline --task galaxy-gravity-v2 >> store/reruns/v62-worker.log 2>&1
log "\"event\": \"galaxy_v2_anchors\", \"store\": \"store-galaxy-v2\""
frozen() { python3 -c "import json,sys;r=[json.loads(l) for l in open(sys.argv[1]) if 'develop_finished' in l];print(r[-1]['submission'] if r else '')" "$1" 2>/dev/null; }
galaxy_sub=$(frozen store/reruns/sol-v62-single-galaxy-v2.log)
if [ -n "$galaxy_sub" ] && [ -f "store-v62/submissions/$galaxy_sub/source.tar" ]; then
    src=$(mktemp -d) && tar -xf "store-v62/submissions/$galaxy_sub/source.tar" -C "$src"
    FRONTIERAM_RUNTIME=$EXP_RT ./bench --store store-galaxy-v2 submit --task galaxy-gravity-v2 --source "$src" \
        --label "gpt-6.1-sol v6.2 single $galaxy_sub" >> store/reruns/v62-worker.log 2>&1
    FRONTIERAM_RUNTIME=$EXP_RT ./bench --store store-galaxy-v2 grade-worker --task galaxy-gravity-v2 --workers 1 --drain >> store/reruns/v62-worker.log 2>&1
    log "\"event\": \"graded\", \"label\": \"sol-single-galaxy-v2\", \"submission\": \"$galaxy_sub\""
fi
species_sub=$(frozen store/reruns/sol-v62-single-species-v2.log)
if [ -n "$species_sub" ]; then
    suite=$(FRONTIERAM_RUNTIME=$EXP_RT ./bench --store store-species-v2 suite --task species-tree-v2 --submission "$species_sub" \
            --shards 4 --slots $SLOTS --isolation shared | last_id)
    FRONTIERAM_RUNTIME=$EXP_RT ./bench --store store-species-v2 grade-worker --task species-tree-v2 --workers 4 --drain >> store/reruns/v62-worker.log 2>&1
    .venv-ops/bin/python scripts/species-v2-suite-score.py --store store-species-v2 --reference "$species_ref" --candidate "$suite" \
        --out results/species-tree-v2/sol-v62-single.json >> store/reruns/v62-score.log 2>&1
    log "\"event\": \"graded\", \"label\": \"sol-single-species-v2\", \"suite\": \"$suite\", \"submission\": \"$species_sub\""
fi

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
grade_conic conic-20261109T065544Z-ff7b81 sol-single-conic
grade_conic "$conic_iter" sol-iterative-conic
log '"event": "done"'
