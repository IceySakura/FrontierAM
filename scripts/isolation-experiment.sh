#!/usr/bin/env bash
# Re-measure grading interference after a change to the grading stack, in a store separate from campaign results.
set -uo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
export PATH="$HOME/.local/bin:$PATH"
store=${1:-store-iso}
dev_load=${DEV_LOAD:-10}
mkdir -p "$store/experiment"
note() { echo "[$(date '+%F %T')] $*" | tee -a "$store/experiment/timeline.log"; }

note "baseline on one slot, nothing else running"
./bench --store "$store" baseline --task exponential > "$store/experiment/baseline.log" 2>&1
note "concurrency 1 and 4, development idle"
./bench --store "$store" interference --levels 1,4 > "$store/experiment/idle.log" 2>&1
note "concurrency 4, $dev_load development slots under memory-bandwidth load"
./bench --store "$store" interference --levels 4 --dev-load "$dev_load" > "$store/experiment/loaded.log" 2>&1
note "done"
