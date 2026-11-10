#!/bin/bash
# Protocol v6.2 under the final rules, then the grading backlog for gpt-6.1-sol and claude-opus-5-5.
# Log: var/stores/store/reruns/v62-final.log
#   1. wait for the exponential-v2 iterative pilot (old prompt) to end
#   2. smoke checks: conic and exponential-v2 iterative, conic capped test (cap shortened for the check only)
#   3. iterative attempts, one at a time: sol conic, sol exponential-v2, opus conic, opus exponential-v2
#   4. exponential-v2 private grading: sol iterative, sol pilot (record), opus single, opus iterative
#   5. reference anchors: species-tree-v2 (sharded), galaxy-gravity-v2
#   6. new-task and operator gradings: galaxy-v2 and species-v2 (sol, opus singles), opus polarization and trajectory
#   7. conic private grading (sol single/iterative, opus single/iterative) with the skip-failed-repeats grader once
#      results/conic/conic-fast-grading-check.json says it is equivalent, else with the unmodified grader
# Single-setting attempts run outside this script (dev slots only); their frozen IDs are read from their logs.
set -uo pipefail
cd "$(dirname "$0")/.."
R=$PWD
LOG=var/stores/store/reruns/v62-final.log
SLOTS=grade-06,grade-14,grade-22,grade-30
KEY=/root/.frontieram/aihub.key
PANEL=suite-exponential-v2-20261109T032227Z-c09381
CONIC_DEV_RT=$R/.runtime-conic-solver1105
EXP_RT=$R/.runtime-recal
P_B=0.7892307692307692
log() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> "$LOG"; }
last_id() { python3 -c "import json,sys;print(json.loads(sys.stdin.readlines()[-1])['id'])"; }
frozen() { python3 -c "import json,sys;r=[json.loads(l) for l in open(sys.argv[1]) if 'develop_finished' in l];print(r[-1]['submission'] if r and r[-1]['status']=='frozen' else '')" "$1" 2>/dev/null; }
wait_frozen() { until grep -q develop_finished "$1" 2>/dev/null; do sleep 60; done; frozen "$1"; }

until [ -f var/stores/store-v62/dev-runs/dev-exponential-v2-20261109T104632Z-517812/develop_result.json ]; do sleep 60; done
pilot=$(frozen var/stores/store/reruns/sol-v62-iterative-exp2.log)
log "\"event\": \"pilot_finished\", \"submission\": \"$pilot\""

smoke() {
    local task=$1 setting=$2 runtime=$3 cap=${4:-}
    FRONTIERAM_V62_TEST_CAP=$cap FRONTIERAM_RUNTIME=$runtime PYTHONPATH=$R nexus/.venv/bin/python scripts/v62-smoke.py \
        "$task" "$setting" 900 "var/stores/store/reruns/v62-smoke/final-$task-$setting-$(date +%H%M%S)" \
        > "var/stores/store/reruns/v62-smoke-final-$task-$setting.log" 2>&1 \
        || { log "\"event\": \"smoke_failed\", \"task\": \"$task\", \"setting\": \"$setting\""; exit 1; }
    log "\"event\": \"smoke_passed\", \"task\": \"$task\", \"setting\": \"$setting\""
}
smoke conic iterative "$CONIC_DEV_RT"
smoke exponential-v2 iterative "$EXP_RT"
smoke conic iterative-cap "$CONIC_DEV_RT" 60

develop_iterative() {
    local model=$1 task=$2 runtime=$3 logfile=$4
    FRONTIERAM_RUNTIME=$runtime ./bench --store var/stores/store-v62 develop --task "$task" --model "$model" --protocol v6.2 \
        --setting iterative --no-grade --key-file $KEY > "$logfile" 2>&1
    frozen "$logfile"
}
declare -A ITER
for model in gpt-6.1-sol claude-opus-5-5; do
    short=${model%%-*}
    for task in conic exponential-v2; do
        runtime=$EXP_RT; [ $task = conic ] && runtime=$CONIC_DEV_RT
        sub=$(develop_iterative "$model" "$task" "$runtime" "var/stores/store/reruns/$short-v62-final-iterative-$task.log")
        ITER[$short-$task]=$sub
        log "\"event\": \"attempt_frozen\", \"model\": \"$model\", \"task\": \"$task\", \"setting\": \"iterative\", \"submission\": \"$sub\""
    done
done

grade_exp2() {
    local submission=$1 label=$2 suite
    [ -n "$submission" ] || { log "\"event\": \"no_submission\", \"label\": \"$label\""; return 0; }
    suite=$(FRONTIERAM_RUNTIME=$EXP_RT ./bench --store var/stores/store-v62 suite --task exponential-v2 --submission "$submission" \
            --shards 4 --slots $SLOTS --isolation shared | last_id) || { log "\"event\": \"queue_failed\", \"label\": \"$label\""; return 1; }
    FRONTIERAM_RUNTIME=$EXP_RT ./bench --store var/stores/store-v62 grade-worker --task exponential-v2 --workers 4 --drain >> var/stores/store/reruns/v62-worker.log 2>&1
    .venv-ops/bin/python scripts/exp2-suite-score.py --store var/stores/store-v62 --panel-store store --panel $PANEL \
        --candidate "$suite" --out "results/runs/v62-$label" >> var/stores/store/reruns/v62-score.log 2>&1
    log "\"event\": \"graded\", \"label\": \"$label\", \"suite\": \"$suite\", \"submission\": \"$submission\""
}
grade_exp2 "${ITER[gpt-exponential-v2]}" sol-iterative-exp2
grade_exp2 "$pilot" sol-iterative-exp2-pilot
grade_exp2 "$(wait_frozen var/stores/store/reruns/opus-v62-single-exp2.log)" opus-single-exp2
grade_exp2 "${ITER[claude-exponential-v2]}" opus-iterative-exp2

until [ -f .runtime-ops/species-tree-v2/lock.json ]; do sleep 60; done
species_ref=$(FRONTIERAM_RUNTIME=$EXP_RT ./bench --store var/stores/store-species-v2 suite --task species-tree-v2 --reference --shards 4 \
              --slots $SLOTS --isolation shared | last_id) || { log '"event": "queue_failed", "label": "species-v2-reference"'; exit 1; }
FRONTIERAM_RUNTIME=$EXP_RT ./bench --store var/stores/store-species-v2 grade-worker --task species-tree-v2 --workers 4 --drain >> var/stores/store/reruns/v62-worker.log 2>&1
.venv-ops/bin/python scripts/species-v2-suite-score.py --store var/stores/store-species-v2 --reference "$species_ref" \
    --out results/species-tree/species-tree-v2/anchors.json >> var/stores/store/reruns/v62-score.log 2>&1
log "\"event\": \"species_v2_anchors\", \"suite\": \"$species_ref\""
FRONTIERAM_RUNTIME=$EXP_RT ./bench --store var/stores/store-galaxy-v2 baseline --task galaxy-gravity-v2 >> var/stores/store/reruns/v62-worker.log 2>&1
log '"event": "galaxy_v2_anchors", "store": "var/stores/store-galaxy-v2"'

grade_galaxy() {
    local logfile=$1 label=$2 sub src
    sub=$(wait_frozen "$logfile")
    [ -n "$sub" ] || { log "\"event\": \"no_submission\", \"label\": \"$label\""; return 0; }
    src=$(mktemp -d) && tar -xf "var/stores/store-v62/submissions/$sub/source.tar" -C "$src"
    FRONTIERAM_RUNTIME=$EXP_RT ./bench --store var/stores/store-galaxy-v2 submit --task galaxy-gravity-v2 --source "$src" --label "$label $sub" \
        >> var/stores/store/reruns/v62-worker.log 2>&1
    FRONTIERAM_RUNTIME=$EXP_RT ./bench --store var/stores/store-galaxy-v2 grade-worker --task galaxy-gravity-v2 --workers 1 --drain >> var/stores/store/reruns/v62-worker.log 2>&1
    log "\"event\": \"graded\", \"label\": \"$label\", \"submission\": \"$sub\""
}
grade_galaxy var/stores/store/reruns/sol-v62-single-galaxy-v2.log sol-single-galaxy-v2
grade_galaxy var/stores/store/reruns/opus-v62-single-galaxy-v2.log opus-single-galaxy-v2

grade_species() {
    local logfile=$1 label=$2 sub suite
    sub=$(wait_frozen "$logfile")
    [ -n "$sub" ] || { log "\"event\": \"no_submission\", \"label\": \"$label\""; return 0; }
    suite=$(FRONTIERAM_RUNTIME=$EXP_RT ./bench --store var/stores/store-species-v2 suite --task species-tree-v2 --submission "$sub" \
            --shards 4 --slots $SLOTS --isolation shared | last_id)
    FRONTIERAM_RUNTIME=$EXP_RT ./bench --store var/stores/store-species-v2 grade-worker --task species-tree-v2 --workers 4 --drain >> var/stores/store/reruns/v62-worker.log 2>&1
    .venv-ops/bin/python scripts/species-v2-suite-score.py --store var/stores/store-species-v2 --reference "$species_ref" --candidate "$suite" \
        --out "results/species-tree/species-tree-v2/$label.json" >> var/stores/store/reruns/v62-score.log 2>&1
    log "\"event\": \"graded\", \"label\": \"$label\", \"suite\": \"$suite\", \"submission\": \"$sub\""
}
grade_species var/stores/store/reruns/sol-v62-single-species-v2.log sol-single-species-v2
grade_species var/stores/store/reruns/opus-v62-single-species-v2.log opus-single-species-v2

grade_operator() {
    local logfile=$1 task=$2 label=$3 sub
    sub=$(wait_frozen "$logfile")
    [ -n "$sub" ] || { log "\"event\": \"no_submission\", \"label\": \"$label\""; return 0; }
    FRONTIERAM_RUNTIME=$EXP_RT ./bench --store var/stores/store grade --submission "$sub" --task "$task" >> var/stores/store/reruns/v62-worker.log 2>&1
    log "\"event\": \"graded\", \"label\": \"$label\", \"submission\": \"$sub\""
}
grade_operator var/stores/store/reruns/opus-v62-single-polarization.log polarization-demosaicking opus-single-polarization
grade_operator var/stores/store/reruns/opus-v62-single-trajectory.log trajectory-covariance opus-single-trajectory

until [ -f results/conic/conic-fast-grading-check.json ]; do sleep 60; done
CONIC_GRADE_RT=$CONIC_DEV_RT
python3 -c "import json,sys;sys.exit(0 if json.load(open('results/conic/conic-fast-grading-check.json')).get('equivalent') else 1)" \
    && CONIC_GRADE_RT=$R/.runtime-conic-solver1105-skip
log "\"event\": \"conic_grader\", \"runtime\": \"$CONIC_GRADE_RT\""
grade_conic() {
    local submission=$1 label=$2 suite
    [ -n "$submission" ] || { log "\"event\": \"no_submission\", \"label\": \"$label\""; return 0; }
    suite=$(FRONTIERAM_RUNTIME=$CONIC_GRADE_RT ./bench --store var/stores/store-v62 suite --task conic --submission "$submission" \
            --shards 4 --slots $SLOTS --isolation shared | last_id) || { log "\"event\": \"queue_failed\", \"label\": \"$label\""; return 1; }
    FRONTIERAM_RUNTIME=$CONIC_GRADE_RT ./bench --store var/stores/store-v62 grade-worker --task conic --workers 4 --drain >> var/stores/store/reruns/v62-worker.log 2>&1
    mkdir -p "results/runs/v62-$label"
    FRONTIERAM_RUNTIME=$CONIC_GRADE_RT ./bench --store var/stores/store-v62 suite-report --suite "$suite" > /dev/null 2>&1
    python3 - "$suite" "$label" "$P_B" <<'PY'
import json, sys
suite, label, p_b = sys.argv[1], sys.argv[2], float(sys.argv[3])
report = json.load(open(f'var/stores/store-v62/suites/{suite}.report.json'))
p = report['success_rate']
score = None if p is None or not report['complete'] else (
    0.0 if p <= 0.9 * p_b else 50 * (p - 0.9 * p_b) / (0.1 * p_b) if p <= p_b else 50 + 50 * (p - p_b) / (1 - p_b))
json.dump({'suite': suite, 'complete': report['complete'], 'successes': report['successes'],
           'cases': report['cases_graded'], 'p': p, 'p_B': p_b, 'score': score, 'groups': report['groups']},
          open(f'results/runs/v62-{label}/score.json', 'w'), indent=1)
PY
    log "\"event\": \"graded\", \"label\": \"$label\", \"suite\": \"$suite\", \"submission\": \"$submission\""
}
grade_conic conic-20261109T065544Z-ff7b81 sol-single-conic
grade_conic "${ITER[gpt-conic]}" sol-iterative-conic
grade_conic "$(wait_frozen var/stores/store/reruns/opus-v62-single-conic.log)" opus-single-conic
grade_conic "${ITER[claude-conic]}" opus-iterative-conic
log '"event": "done"'
