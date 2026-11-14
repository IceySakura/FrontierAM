#!/usr/bin/env bash
# Protocol v6.3 validation round: reference anchors for the v6.3 versions, then private grading of every finished
# gpt-6.1-sol attempt (scripts/rounds/v63-sol.sh). Rerun-safe through var/stores/store/reruns/v63-grade.log.
# Held-out grading is capped at 1 h per job (bench suite --cap-s / develop enqueue); suites use 7 shards (iterative
# tests take priority leases, no slot is held back). Each frozen submission is graded in the background as soon as it
# appears (polled every 30 s). Scores land in tasks/<family>/solutions/v6.3/<model>-<setting>-<n>/score.json.
set -uo pipefail
cd "$(dirname "$0")/../.."
R=$PWD
STORE=var/stores/store-v63
LOG=var/stores/store/reruns/v63-grade.log
N=${V63_ATTEMPT:-1}
MODEL=${V63_MODEL:-gpt-6.1-sol}   # grade-log labels: <setting>-<task>-<n> for sol, opus-<setting>-<task>-<n> for opus
case $MODEL in gpt-6.1-sol) TAG=sol PFX=;; claude-opus-5-5) TAG=opus PFX=opus-;; qwen3.8-max|qwen3.8-max-0902) TAG=qwen PFX=qwen-;; hy4) TAG=hy4 PFX=hy4-;; esac
EXP_RT=$R/.runtime-recal
CONIC_RT=$R/.runtime-conic-v3
P_B=0.7892307692307692          # conic v3 keeps every calibrated limit; the reference solves the same 513 of 650
PANEL=suite-exponential-v2-20261109T032227Z-c09381   # exponential v3: same cases and panel as v2, covered under v3 limits
CAP=3600
log() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> "$LOG"; }
done_event() { grep -q "\"event\": \"$1\", \"label\": \"$2\"" "$LOG" 2>/dev/null; }
value() { grep "\"event\": \"$1\", \"label\": \"$2\"" "$LOG" | tail -1 | python3 -c "import json,sys;print(json.loads(sys.stdin.read())['$3'])"; }
last_id() { python3 -c "import json,sys;print(json.loads(sys.stdin.readlines()[-1])['id'])"; }
family() { case $1 in conic) echo conic;; exponential-v3) echo exponential;; galaxy-gravity-v3|galaxy-gravity-v4) echo galaxy-gravity;;
    polarization-demosaicking-v2) echo polarization-demosaicking;; species-tree-v3) echo species-tree;;
    trajectory-covariance-v2) echo trajectory-covariance;; ihvp-sequence|ihvp-sequence-v2|ihvp-sequence-v3) echo ihvp-sequence;; esac; }
runtime() { [ "$1" = conic ] && echo "$CONIC_RT" || echo "$EXP_RT"; }
# frozen submission ID; '-' for an invalid submission (scores 0); '' while running or after an infrastructure failure
# (status unmeasured: the attempt is rerun, its log renamed, never graded)
frozen() { python3 -c "import json,sys;r=[json.loads(l) for l in open(sys.argv[1]) if 'develop_finished' in l];print(r[-1]['submission'] if r and r[-1]['status']=='frozen' else ('-' if r and r[-1]['status']=='invalid_submission' else ''))" "$1" 2>/dev/null; }

# Full concurrency since 2026-11-12 (docs/platform/CONCURRENCY.md section 7): no grading waits for development.
TIMED=""
attempts_running() { pgrep -f "scripts/rounds/v63-sol.sh|v63-singles-lane" > /dev/null; }
isolation() { case " $TIMED " in *" $1 "*) echo dev_idle;; *) echo shared;; esac; }

suite() {  # task, submission|--reference -> suite id; grades it now
    local task=$1 target=$2 id
    id=$(FRONTIERAM_RUNTIME=$(runtime "$task") ./bench --store $STORE suite --task "$task" $target --shards 7 \
         --isolation "$(isolation "$task")" --cap-s $CAP | last_id) || return 1
    FRONTIERAM_RUNTIME=$(runtime "$task") ./bench --store $STORE grade-worker --task "$task" --workers 7 --drain \
        >> var/stores/store/reruns/v63-worker.log 2>&1
    # another loop's worker for the same task may still be grading some of this suite's shards
    # a shard that is in no queue and not done has failed: give up this suite (the cell is regraded on a later pass)
    local state
    while state=$(python3 -c "
import json,os
r=json.load(open('$STORE/suites/$id.json')); q='$STORE/jobs/'
s=[next((k for k in ('done','pending','running','held') if os.path.exists(q+k+'/'+x['job']+'.json')),'lost') for x in r['shards']]
print('lost' if 'lost' in s else 'done' if all(k=='done' for k in s) else 'wait')"); [ "$state" = wait ]; do
        sleep 30
    done
    [ "$state" = done ] || { echo "{\"suite\": \"$id\", \"event\": \"shard_lost\"}" >&2; return 1; }
    echo "$id"
}

anchors() {
    local task id
    for task in species-tree-v3 trajectory-covariance-v2 galaxy-gravity-v4; do
        done_event anchor "$task" && continue
        # galaxy-gravity-v4 is anchored only under the per-case limits calibrated on this machine (prepare_galaxy_v4.py)
        [ "$task" = galaxy-gravity-v4 ] && ! grep -q '"calibrated": true' .runtime-ops/galaxy-gravity-v4/lock.json && continue
        # several loops may reach the same anchor; the first measures it, the others wait and reuse it
        ( flock 8; done_event anchor "$task" || { id=$(suite "$task" --reference) && log "\"event\": \"anchor\", \"label\": \"$task\", \"suite\": \"$id\""; } ) \
            8> "var/stores/store/reruns/anchor-$task.lock"
    done
    for task in galaxy-gravity-v3 polarization-demosaicking-v2 ihvp-sequence ihvp-sequence-v3; do
        done_event anchor "$task" && continue
        FRONTIERAM_RUNTIME=$EXP_RT ./bench --store $STORE baseline --task "$task" >> var/stores/store/reruns/v63-worker.log 2>&1 \
            && log "\"event\": \"anchor\", \"label\": \"$task\""
    done
}

grade() {  # setting, task
    local setting=$1 task=$2 label="$PFX$1-$2-$N" attempt_log="var/stores/store/reruns/v63-$TAG-$1-$2-$N.log" sub id out
    done_event graded "$label" && return 0
    sub=$(frozen "$attempt_log")
    [ -z "$sub" ] && return 1                       # attempt still running
    case " $TIMED " in *" $task "*) attempts_running && return 1;; esac
    out=tasks/$(family "$task")/solutions/v6.3/$(case $task in ihvp-sequence-v2) echo v2/;; ihvp-sequence-v3) echo v3/;; galaxy-gravity-v4) echo v4/;; esac)$MODEL-$setting-$N
    mkdir -p "$out"
    if [ "$sub" = - ]; then                          # no valid frozen submission: invalid (0) or unmeasured
        run=$(python3 -c "import json,sys;print([json.loads(l) for l in open(sys.argv[1]) if 'develop_finished' in l][-1]['run'])" \
              "$attempt_log")
        cp "$STORE/dev-runs/$run/develop_result.json" "$out/develop_result.json" 2>/dev/null
        log "\"event\": \"graded\", \"label\": \"$label\", \"submission\": null"
        scripts/rounds/publish-now.sh "$label" &
        return 0
    fi
    case $task in
        conic)
            id=$(suite conic "--submission $sub") || return 1
            FRONTIERAM_RUNTIME=$CONIC_RT ./bench --store $STORE suite-report --suite "$id" > /dev/null 2>&1
            python3 - "$id" "$out" "$P_B" <<'PY'
import json, sys
suite, out, p_b = sys.argv[1], sys.argv[2], float(sys.argv[3])
report = json.load(open(f'var/stores/store-v63/suites/{suite}.report.json'))
p = report['success_rate']
score = None if p is None or not report['complete'] else (
    0.0 if p <= 0.9 * p_b else 50 * (p - 0.9 * p_b) / (0.1 * p_b) if p <= p_b else 50 + 50 * (p - p_b) / (1 - p_b))
json.dump({'suite': suite, 'complete': report['complete'], 'successes': report['successes'], 'cases': report['cases_graded'],
           'p': p, 'p_B': p_b, 'score': score, 'groups': report['groups']}, open(f'{out}/score.json', 'w'), indent=1)
PY
            ;;
        exponential-v3)
            id=$(suite exponential-v3 "--submission $sub") || return 1
            .venv-ops/bin/python scripts/grade/exp2-suite-score.py --task exponential-v3 --store $STORE --panel-store var/stores/store \
                --panel $PANEL --candidate "$id" --out "$out" \
                >> var/stores/store/reruns/v63-score.log 2>&1 ;;
        galaxy-gravity-v4)
            done_event anchor "$task" || return 1
            id=$(suite "$task" "--submission $sub") || return 1
            .venv-ops/bin/python scripts/grade/galaxy-v4-suite-score.py --store $STORE --reference "$(value anchor "$task" suite)" \
                --candidate "$id" --out "$out/score.json" >> var/stores/store/reruns/v63-score.log 2>&1 ;;
        species-tree-v3|trajectory-covariance-v2)
            done_event anchor "$task" || return 1
            id=$(suite "$task" "--submission $sub") || return 1
            .venv-ops/bin/python scripts/grade/$([ "$task" = species-tree-v3 ] && echo species-v2 || echo trajectory)-suite-score.py \
                --store $STORE --reference "$(value anchor "$task" suite)" --candidate "$id" --out "$out/score.json" \
                >> var/stores/store/reruns/v63-score.log 2>&1 ;;
        *)
            done_event anchor "$task" || return 1
            job=$(FRONTIERAM_RUNTIME=$EXP_RT ./bench --store $STORE grade --submission "$sub" --task "$task" 2>&1 | tail -1)
            python3 - "$STORE" "$sub" "$out" <<'PY'
import json, sys
from pathlib import Path
store, sub, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
done = [json.loads(p.read_text()) for p in (store / 'jobs/done').glob('*.json')]
rows = [d for d in done if d.get('submission') == sub and d.get('kind') == 'candidate']
record = rows[-1] if rows else {'status': 'unmeasured'}
metrics = store / 'results' / record.get('id', '-') / 'grade/metrics.json'
if metrics.exists():  # raw error and the run details of the candidate part
    record['candidate'] = json.loads(metrics.read_text())['parts'].get('candidate')
json.dump(record, open(f'{out}/score.json', 'w'), indent=1)
PY
            id=job ;;
    esac
    log "\"event\": \"graded\", \"label\": \"$label\", \"submission\": \"$sub\", \"suite\": \"$id\""
    scripts/rounds/publish-now.sh "$label" &
}

TASKS=${V63_TASKS:-"conic exponential-v3 galaxy-gravity-v3 galaxy-gravity-v4 polarization-demosaicking-v2 species-tree-v3 trajectory-covariance-v2 ihvp-sequence ihvp-sequence-v2 ihvp-sequence-v3"}
INFLIGHT=var/stores/store/reruns/v63-grade-inflight-$TAG-$(echo "$TASKS" | md5sum | cut -c1-8)   # per loop: one marker per background grading
rm -rf "$INFLIGHT" && mkdir -p "$INFLIGHT"
while :; do
    anchors
    left=0
    for setting in single iterative; do
        for task in $TASKS; do
            label="$PFX$setting-$task-$N"
            done_event graded "$label" && continue
            # an attempt with no log while no attempt runner is running was not requested (V63_SETTINGS)
            [ -e "var/stores/store/reruns/v63-$TAG-$setting-$task-$N.log" ] || attempts_running || continue
            left=$((left + 1))
            [ -e "$INFLIGHT/$label" ] && continue
            [ -z "$(frozen "var/stores/store/reruns/v63-$TAG-$setting-$task-$N.log")" ] && continue
            touch "$INFLIGHT/$label"
            # one grading per task at a time: a suite's grade-worker drains the task's shared queue
            ( flock 9; grade "$setting" "$task"; rm -f "$INFLIGHT/$label" ) 9> "$INFLIGHT/$task.lock" &
        done
    done
    [ "$left" = 0 ] && break
    sleep 30
done
wait
log "\"event\": \"all_graded\""
