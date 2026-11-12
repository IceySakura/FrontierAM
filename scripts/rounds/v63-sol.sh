#!/usr/bin/env bash
# Protocol v6.3 validation round (results/PROTOCOL_V63.md): gpt-6.1-sol on the v6.3 task versions (ihvp-sequence added 11-11), iterative (4 h)
# and single (1 h), at most V63_CONCURRENCY attempts at a time (default 3: the model's API rate limit); iterative first.
# Each attempt leases its own dev slot; no grading here (scripts/rounds/v63-grade.sh grades finished attempts).
# Rerun-safe: an attempt whose log already exists (started or finished) is skipped; infrastructure failures are
# rerun by renaming their log.
# Logs: var/stores/store/reruns/v63-sol-<setting>-<task>-<n>.log
set -uo pipefail
cd "$(dirname "$0")/../.."
R=$PWD
KEY=/root/.frontieram/aihub.key
STORE=var/stores/store-v63
MAX=${V63_CONCURRENCY:-3}
N=${V63_ATTEMPT:-1}
TASKS="conic exponential-v3 galaxy-gravity-v3 polarization-demosaicking-v2 species-tree-v3 trajectory-covariance-v2 ihvp-sequence ihvp-sequence-v3"

attempt() {
    local setting=$1 task=$2 log=var/stores/store/reruns/v63-sol-$1-$2-$N.log runtime=$R/.runtime-recal
    grep -q develop_ "$log" 2>/dev/null && return 0      # finished, or started by an earlier pass (still running)
    [ "$task" = conic ] && runtime=$R/.runtime-conic-v3
    FRONTIERAM_RUNTIME=$runtime ./bench --store $STORE develop --task "$task" --model gpt-6.1-sol --protocol v6.3 \
        --setting "$setting" --no-grade --key-file $KEY > "$log" 2>&1 < /dev/null
}
export -f attempt
export R KEY STORE N
{ for t in $TASKS; do echo "iterative $t"; done; for t in $TASKS; do echo "single $t"; done; } |
    xargs -P "$MAX" -L 1 bash -c 'attempt "$0" "$1"'
