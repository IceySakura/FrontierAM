#!/usr/bin/env bash
# Round 5 (results/PROTOCOL_V62.md): v6.2 single-setting attempts, all started together on development slots (each
# develop run leases its own dev slot; no grading here). Empty cells: gpt-6.1-sol polarization and trajectory.
# Repeats (2 each): claude-opus-5-5 conic, gpt-6.1-sol exponential-v2, claude-opus-5-5 exponential-v2.
# Logs: var/stores/store/reruns/r5-<model>-single-<task>-<n>.log (develop_finished carries the frozen submission).
set -uo pipefail
cd "$(dirname "$0")/.."
R=$PWD
KEY=/root/.frontieram/aihub.key
CONIC_RT=$R/.runtime-conic-solver1105
EXP_RT=$R/.runtime-recal

start() {
    local store=$1 runtime=$2 task=$3 model=$4 n=$5
    local short=${model%%-*}
    FRONTIERAM_RUNTIME=$runtime nohup setsid ./bench --store "$store" develop --task "$task" --model "$model" \
        --protocol v6.2 --setting single --no-grade --key-file $KEY \
        > "var/stores/store/reruns/r5-$short-single-$task-$n.log" 2>&1 < /dev/null &
    sleep 5
}

start store "$EXP_RT" polarization-demosaicking gpt-6.1-sol 1
start store "$EXP_RT" trajectory-covariance gpt-6.1-sol 1
for n in 2 3; do
    start var/stores/store-v62 "$CONIC_RT" conic claude-opus-5-5 $n
    start var/stores/store-v62 "$EXP_RT" exponential-v2 gpt-6.1-sol $n
    start var/stores/store-v62 "$EXP_RT" exponential-v2 claude-opus-5-5 $n
done
wait
