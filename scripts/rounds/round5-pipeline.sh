#!/usr/bin/env bash
# Round 5 (results/PROTOCOL_V62.md): grade the single-setting attempts, run the iterative repeats one at a time, grade
# them, then the jobs that need a fully quiet machine. Log: var/stores/store/reruns/r5-pipeline.log; scores in tasks/<family>/solutions/v6.2/<model>-<setting>-<n>/.
#   1. private grading: opus conic single x2, opus exp2 single x2, sol exp2 single x2, sol trajectory single
#   2. iterative attempts (each holds the machine-level iterative lock): opus conic x2, opus exp2 x2
#   3. private grading of the four iterative attempts
#   4. quiet-isolation jobs: sol polarization single, polarization warm-up check, second galaxy-v2 reference rerun
set -uo pipefail
cd "$(dirname "$0")/../.."
R=$PWD
LOG=var/stores/store/reruns/r5-pipeline.log
SLOTS=grade-06,grade-14,grade-22,grade-30
KEY=/root/.frontieram/aihub.key
PANEL=suite-exponential-v2-20261109T032227Z-c09381
CONIC_DEV_RT=$R/.runtime-conic-solver1105
CONIC_GRADE_RT=$R/.runtime-conic-solver1105-skip
EXP_RT=$R/.runtime-recal
P_B=0.7892307692307692
log() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> "$LOG"; }
last_id() { python3 -c "import json,sys;print(json.loads(sys.stdin.readlines()[-1])['id'])"; }
outdir() {  # pipeline label -> attempt directory of the task
    local n=${1##*-}
    case $1 in
        r5-opus-single-exp2-*) echo "tasks/exponential/solutions/v6.2/claude-opus-5-5-single-$n" ;;
        r5-sol-single-exp2-*) echo "tasks/exponential/solutions/v6.2/gpt-6.1-sol-single-$n" ;;
        r5-opus-iterative-exp2-*) echo "tasks/exponential/solutions/v6.2/claude-opus-5-5-iterative-$n" ;;
        r5-opus-single-conic-*) echo "tasks/conic/solutions/v6.2/claude-opus-5-5-single-$n" ;;
        r5-opus-iterative-conic-*) echo "tasks/conic/solutions/v6.2/claude-opus-5-5-iterative-$n" ;;
        *) echo "unknown label $1" >&2; return 1 ;;
    esac
}
graded() { grep -q "\"graded\", \"label\": \"$1\"" "$LOG" 2>/dev/null; }
frozen() { python3 -c "import json,sys;r=[json.loads(l) for l in open(sys.argv[1]) if 'develop_finished' in l];print(r[-1]['submission'] if r and r[-1]['status']=='frozen' else '')" "$1" 2>/dev/null; }

grade_exp2() {
    local submission=$1 label=$2 suite
    graded "$label" && return 0
    suite=$(FRONTIERAM_RUNTIME=$EXP_RT ./bench --store var/stores/store-v62 suite --task exponential-v2 --submission "$submission" \
            --shards 4 --slots $SLOTS --isolation shared | last_id) || { log "\"event\": \"queue_failed\", \"label\": \"$label\""; return 1; }
    FRONTIERAM_RUNTIME=$EXP_RT ./bench --store var/stores/store-v62 grade-worker --task exponential-v2 --workers 4 --drain >> var/stores/store/reruns/r5-worker.log 2>&1
    .venv-ops/bin/python scripts/grade/exp2-suite-score.py --store var/stores/store-v62 --panel-store store --panel $PANEL \
        --candidate "$suite" --out "$(outdir "$label")" >> var/stores/store/reruns/r5-score.log 2>&1
    log "\"event\": \"graded\", \"label\": \"$label\", \"suite\": \"$suite\", \"submission\": \"$submission\""
}

grade_conic() {
    local submission=$1 label=$2 suite
    graded "$label" && return 0
    suite=$(FRONTIERAM_RUNTIME=$CONIC_GRADE_RT ./bench --store var/stores/store-v62 suite --task conic --submission "$submission" \
            --shards 4 --slots $SLOTS --isolation shared | last_id) || { log "\"event\": \"queue_failed\", \"label\": \"$label\""; return 1; }
    FRONTIERAM_RUNTIME=$CONIC_GRADE_RT ./bench --store var/stores/store-v62 grade-worker --task conic --workers 4 --drain >> var/stores/store/reruns/r5-worker.log 2>&1
    mkdir -p "$(outdir "$label")"
    FRONTIERAM_RUNTIME=$CONIC_GRADE_RT ./bench --store var/stores/store-v62 suite-report --suite "$suite" > /dev/null 2>&1
    python3 - "$suite" "$(outdir "$label")" "$P_B" <<'PY'
import json, sys
suite, out, p_b = sys.argv[1], sys.argv[2], float(sys.argv[3])
report = json.load(open(f'var/stores/store-v62/suites/{suite}.report.json'))
p = report['success_rate']
score = None if p is None or not report['complete'] else (
    0.0 if p <= 0.9 * p_b else 50 * (p - 0.9 * p_b) / (0.1 * p_b) if p <= p_b else 50 + 50 * (p - p_b) / (1 - p_b))
json.dump({'suite': suite, 'complete': report['complete'], 'successes': report['successes'],
           'cases': report['cases_graded'], 'p': p, 'p_B': p_b, 'score': score, 'groups': report['groups']},
          open(f'{out}/score.json', 'w'), indent=1)
PY
    log "\"event\": \"graded\", \"label\": \"$label\", \"suite\": \"$suite\", \"submission\": \"$submission\""
}

grade_operator() {
    local submission=$1 task=$2 label=$3
    graded "$label" && return 0
    FRONTIERAM_RUNTIME=$EXP_RT ./bench --store var/stores/store grade --submission "$submission" --task "$task" >> var/stores/store/reruns/r5-worker.log 2>&1
    log "\"event\": \"graded\", \"label\": \"$label\", \"submission\": \"$submission\""
}

develop_iterative() {
    local task=$1 runtime=$2 logfile=$3
    FRONTIERAM_RUNTIME=$runtime ./bench --store var/stores/store-v62 develop --task "$task" --model claude-opus-5-5 --protocol v6.2 \
        --setting iterative --no-grade --key-file $KEY > "$logfile" 2>&1
    frozen "$logfile"
}

log '"event": "start"'  # rerun-safe: graded labels are skipped
grade_exp2 exponential-v2-20261110T041026Z-22d0e1 r5-opus-single-exp2-2
grade_exp2 exponential-v2-20261110T043349Z-380434 r5-opus-single-exp2-3
grade_exp2 exponential-v2-20261110T031548Z-80934e r5-sol-single-exp2-2
grade_exp2 exponential-v2-20261110T053617Z-d8bb77 r5-sol-single-exp2-3
grade_conic conic-20261110T044139Z-e1ef6c r5-opus-single-conic-2
grade_conic conic-20261110T041406Z-762c81 r5-opus-single-conic-3
grade_operator trajectory-covariance-20261110T035541Z-b16076 trajectory-covariance r5-sol-single-trajectory

declare -A ITER
if [ -e /run/frontieram/hold-iterative ]; then
    log "\"event\": \"iterative_held\", \"reason\": \"protocol under discussion\""
else
    for n in 2 3; do
        for task in conic exponential-v2; do
            runtime=$EXP_RT; [ $task = conic ] && runtime=$CONIC_DEV_RT
            sub=$(develop_iterative "$task" "$runtime" "var/stores/store/reruns/r5-claude-iterative-$task-$n.log")
            ITER[$task-$n]=$sub
            log "\"event\": \"attempt_frozen\", \"model\": \"claude-opus-5-5\", \"task\": \"$task\", \"setting\": \"iterative\", \"n\": $n, \"submission\": \"$sub\""
        done
    done
    for n in 2 3; do
        [ -n "${ITER[exponential-v2-$n]}" ] && grade_exp2 "${ITER[exponential-v2-$n]}" "r5-opus-iterative-exp2-$n"
        [ -n "${ITER[conic-$n]}" ] && grade_conic "${ITER[conic-$n]}" "r5-opus-iterative-conic-$n"
    done
fi

grade_operator polarization-demosaicking-20261110T035607Z-9748d5 polarization-demosaicking r5-sol-single-polarization
.venv-ops/bin/python scripts/analyze/polar-warmup-check.py polarization-demosaicking-20261109T124343Z-553533 \
    tasks/polarization-demosaicking/analysis/polarization-first-scene.json >> var/stores/store/reruns/r5-worker.log 2>&1
log '"event": "polarization_warmup_quiet_check"'
FRONTIERAM_RUNTIME=$EXP_RT ./bench --store var/stores/store-galaxy-v2-var2 baseline --task galaxy-gravity-v2 >> var/stores/store/reruns/r5-worker.log 2>&1
log '"event": "galaxy_v2_reference_rerun"'
log '"event": "done"'
