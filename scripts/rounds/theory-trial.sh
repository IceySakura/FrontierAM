#!/usr/bin/env bash
# Theory-only trial (user decision 2026-11-27; tasks/blackbox-cubature/analysis/DUAL_PLAN.md section 9.15): one
# single-setting attempt with a 6-hour limit in the v6.3 harness. --development-seconds marks the submission as a
# validation run, so it never enters the leaderboard. Grading is the Lean check (scripts/grade/lean-comparator.py) and
# the theory review (analysis/THEORY_REVIEW.md); there is no program to grade.
# Usage: theory-trial.sh TASK [MODEL] [N]
# log: var/stores/store/reruns/theory-<task>-<tag>-<n>.log
set -uo pipefail
cd "$(dirname "$0")/../.."
TASK=${1:?usage: theory-trial.sh TASK [MODEL] [N]}
MODEL=${2:-gpt-6.2-sol}
N=${3:-1}
case $MODEL in gpt-6.2-sol) TAG=sol62;; claude-opus-5-5) TAG=opus;; *) echo "unknown model $MODEL" >&2; exit 2;; esac
case $TASK in blackbox-cubature-theory-v1|decentralized-sum-of-nonconvex-theory-v1|blackbox-cubature-theory-v2|decentralized-sum-of-nonconvex-theory-v2) ;; *) echo "unknown theory task $TASK" >&2; exit 2;; esac
LOG=var/stores/store/reruns/theory-$TASK-$TAG-$N.log
[ -e "$LOG" ] && { echo "$LOG exists" >&2; exit 1; }
FRONTIERAM_RUNTIME=$PWD/.runtime-recal ./bench --store var/stores/store-v63 develop --task "$TASK" \
    --model "$MODEL" --protocol v6.3 --setting single --development-seconds 21600 --no-grade \
    --key-file /root/.frontieram/aihub.key > "$LOG" 2>&1 < /dev/null
