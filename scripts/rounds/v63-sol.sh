#!/usr/bin/env bash
# Protocol v6.3 validation round (results/PROTOCOL_V63.md): gpt-6.1-sol on the v6.3 task versions (ihvp-sequence added 11-11), iterative (4 h)
# and single (1 h), at most V63_CONCURRENCY attempts at a time (default 3: the model's API rate limit); singles first.
# Each attempt leases its own dev slot; no grading here (scripts/rounds/v63-grade.sh grades finished attempts).
# Rerun-safe: an attempt whose log already exists (started or finished) is skipped; infrastructure failures are
# rerun by renaming their log.
# Logs: var/stores/store/reruns/v63-<tag>-<setting>-<task>-<n>.log (tag: sol, opus).
# V63_MODEL picks the model (default gpt-6.1-sol); V63_TASKS overrides the task list, V63_SETTINGS the settings.
set -uo pipefail
cd "$(dirname "$0")/../.."
R=$PWD
KEY=/root/.frontieram/aihub.key
STORE=var/stores/store-v63
MAX=${V63_CONCURRENCY:-3}
N=${V63_ATTEMPT:-1}
MODEL=${V63_MODEL:-gpt-6.1-sol}
case $MODEL in gpt-6.1-sol) TAG=sol;; gpt-6.2-sol) TAG=sol62;; gpt-6-astra) TAG=astra;; claude-opus-5-5) TAG=opus;; qwen3.8-max|qwen3.8-max-0902) TAG=qwen;; hy4) TAG=hy4;; intern-s2) TAG=intern;; qwen4-max) TAG=qwen4;; glm-5.4) TAG=glm;; minimax-m3.1-flash) TAG=minimax;; *) echo "unknown model $MODEL" >&2; exit 2;; esac
TASKS=${V63_TASKS:-"conic exponential-v3 galaxy-gravity-v3 galaxy-gravity-v4 polarization-demosaicking-v2 species-tree-v3 trajectory-covariance-v2 ihvp-sequence ihvp-sequence-v3"}
SETTINGS=${V63_SETTINGS:-"single iterative"}

attempt() {
    local setting=$1 task=$2 log=var/stores/store/reruns/v63-$TAG-$1-$2-$N.log runtime=$R/.runtime-recal
    grep -q develop_ "$log" 2>/dev/null && return 0      # finished, or started by an earlier pass (still running)
    [ "$task" = conic ] && runtime=$R/.runtime-conic-v3
    FRONTIERAM_RUNTIME=$runtime ./bench --store $STORE develop --task "$task" --model $MODEL --protocol v6.3 \
        --setting "$setting" --no-grade --key-file $KEY > "$log" 2>&1 < /dev/null
}
export -f attempt
export R KEY STORE N MODEL TAG
# singles first: they free their slots within an hour and keep the grade slots busy while the iteratives run
{ for s in $SETTINGS; do for t in $TASKS; do echo "$s $t"; done; done; } |
    xargs -P "$MAX" -L 1 bash -c 'attempt "$0" "$1"'
