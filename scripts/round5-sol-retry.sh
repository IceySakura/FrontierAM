#!/usr/bin/env bash
# Round 5: rerun the three gpt-6.1-sol single attempts that ended on 11-10 ~11:42 with upstream "429 Too Many Requests"
# (infrastructure failure, recorded as unmeasured, not graded). At most two sol attempts at a time; the first round had
# four concurrent sol sessions. Logs: store/reruns/r5-gpt-single-<task>-<n>-retry.log.
set -uo pipefail
cd "$(dirname "$0")/.."
R=$PWD
KEY=/root/.frontieram/aihub.key

run() {
    local store=$1 task=$2 n=$3
    FRONTIERAM_RUNTIME=$R/.runtime-recal ./bench --store "$store" develop --task "$task" --model gpt-6.1-sol \
        --protocol v6.2 --setting single --no-grade --key-file $KEY > "store/reruns/r5-gpt-single-$task-$n-retry.log" 2>&1
}

run store polarization-demosaicking 1 &
sleep 30
run store trajectory-covariance 1 &
wait -n
run store-v62 exponential-v2 3
wait
