#!/bin/sh
set -eu
if [ "$#" -ne 2 ]; then
    printf '%s\n' 'Usage: run.sh INPUT OUTPUT' >&2
    exit 2
fi
cd "$(dirname "$0")"
solver="$PWD/species-tree"
input=$1
output=$2
if [ -d "$output" ]; then
    printf '%s\n' 'OUTPUT must be a file path' >&2
    exit 2
fi
work=$(mktemp -d "${TMPDIR:-/tmp}/species-tree.XXXXXX")
trap 'rm -rf "$work"' 0
trap 'exit 1' 1 2 15

infer() {
    name=$1
    limit=$2
    shift 2
    env DIST_MODE=3 DIST_PARAM=0.5 BIONJ=0 UPGMA=0 FARRIS=0 DIST_QUANTILE=-1 \
        NNI_ROUNDS=30 SPR_ROUNDS=2 SPR_RADIUS=4 BME_ROUNDS=0 \
        LAMBDA=0 SUP_POWER=1 SUP_FLOOR=0.1 WEIGHT_MODE=0 MAX_WEIGHT=0 \
        LENGTH_RATE=1 LENGTH_MIX=0.5 RELATIVE_DECAY=0 SUP_DECAY=0 \
        ZERO_COLLAPSE=0 MISSING_SUPPORT=1 DP_MODE=0 \
        BOOT_REPS=0 BOOT_START=0 BOOT_NNI=12 BOOT_SPR=0 BOOT_TYPE=0 \
        BOOT_PRIOR=1 BOOT_CACHE=1 PRECOMPUTE_DIST=1 BOOT_DP=0 \
        BOOT_DP_GENES=0 BOOT_DP_REPS=1000 BOOT_WARMUP=30 BOOT_POOL_EXTRA=0 \
        BOOT_YULE=0 BOOT_BALANCE=0 BOOT_BALANCE_ROUNDS=3 BOOT_QUARTET=0 \
        FINAL_BALANCE=0 BALANCE_ROUNDS=4 BALANCE_SPR=0 DISTANCE_PRIOR=0 \
        DEBUG_SCORE=0 TIME_LIMIT="$limit" "$@" \
        "$solver" "$input" "$work/$name.tre"
}

set --
if infer T 40 SUP_DECAY=0.05 FINAL_BALANCE=0.75; then
    set -- "$@" "$work/T.tre"
fi
if infer U 40 SUP_DECAY=0.1 FINAL_BALANCE=0.75; then
    set -- "$@" "$work/U.tre"
fi
if infer D 40 WEIGHT_MODE=1 SUP_FLOOR=0; then
    set -- "$@" "$work/D.tre"
fi
if [ "$#" -eq 0 ]; then
    exit 1
fi
if ! "$solver" --consensus "$output" "$@"; then
    mkdir -p "$(dirname "$output")"
    cp "$1" "$output"
fi
