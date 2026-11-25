#!/usr/bin/env bash
# Protocol v6.3 attempt lane: runs the attempts listed in a queue file ("MODEL SETTING TASK N" per line, in order) with
# at most MAX at a time. Each attempt leases its own dev slot (bench develop waits for one), so several lanes can share
# the machine; an attempt is claimed atomically (mkdir), so lanes reading the same queue never start it twice.
# Logs: var/stores/store/reruns/v63-<tag>-<setting>-<task>-<n>.log, as scripts/rounds/v63-sol.sh; grading is separate
# (scripts/rounds/v63-grade.sh).
# CLAIM names the claim marker (default claim): a dedicated lane for one model uses its own marker, after the
# shared lanes' markers for that model's attempts were created in advance so the shared lanes skip them.
# Usage: [CLAIM=name] v63-lane.sh QUEUE [MAX]
set -uo pipefail
cd "$(dirname "$0")/../.."
R=$PWD
QUEUE=$1
MAX=${2:-2}
CLAIM=${CLAIM:-claim}
STORE=var/stores/store-v63

attempt() {
    local model=$1 setting=$2 task=$3 n=$4 tag key=/root/.frontieram/aihub.key runtime=$R/.runtime-recal
    case $model in gpt-6.1-sol) tag=sol;; gpt-6.2-sol) tag=sol62;; gpt-6-astra) tag=astra;; claude-opus-5-5) tag=opus;; qwen3.8-max|qwen3.8-max-0902) tag=qwen;; hy4) tag=hy4;;
        intern-s2) tag=intern;; qwen4-max) tag=qwen4;; minimax-m3.1-flash) tag=minimax;; *) echo "unknown model $model" >&2; return 2;; esac
    local log=var/stores/store/reruns/v63-$tag-$setting-$task-$n.log
    mkdir "$log.$CLAIM" 2>/dev/null || return 0        # another lane took it, or an earlier pass ran it
    grep -q develop_ "$log" 2>/dev/null && return 0
    [ "$task" = conic ] && runtime=$R/.runtime-conic-v3
    FRONTIERAM_RUNTIME=$runtime ./bench --store $STORE develop --task "$task" --model "$model" --protocol v6.3 \
        --setting "$setting" --no-grade --key-file $key > "$log" 2>&1 < /dev/null
}
export -f attempt
export R STORE CLAIM
grep -v '^\s*#' "$QUEUE" | grep -v '^\s*$' | xargs -P "$MAX" -L 1 bash -c 'attempt "$0" "$1" "$2" "$3"'
