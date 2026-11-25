#!/usr/bin/env bash
# blackbox-cubature v3 theory + experiment trial (tasks/blackbox-cubature/analysis/DUAL_PLAN.md, M4): one iterative
# attempt with a 6-hour limit in the v6.3 harness. --development-seconds marks the submission as a validation run, so
# it never enters the leaderboard. Grading is separate: a private suite against the locally measured reference suite
# (scripts/grade/cubature-suite-score.py --task blackbox-cubature-v3).
# Usage: v3-dual-trial.sh [MODEL] [N]     log: var/stores/store/reruns/v3trial-<tag>-iterative-<n>.log
set -uo pipefail
cd "$(dirname "$0")/../.."
MODEL=${1:-gpt-6.2-sol}
N=${2:-1}
case $MODEL in gpt-6.2-sol) TAG=sol62;; gpt-6.1-sol) TAG=sol;; claude-opus-5-5) TAG=opus;; *) echo "unknown model $MODEL" >&2; exit 2;; esac
LOG=var/stores/store/reruns/v3trial-$TAG-iterative-$N.log
[ -e "$LOG" ] && { echo "$LOG exists" >&2; exit 1; }
FRONTIERAM_RUNTIME=$PWD/.runtime-recal ./bench --store var/stores/store-v63 develop --task blackbox-cubature-v3 \
    --model "$MODEL" --protocol v6.3 --setting iterative --development-seconds 21600 --no-grade \
    --key-file /root/.frontieram/aihub.key > "$LOG" 2>&1 < /dev/null
