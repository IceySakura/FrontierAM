#!/usr/bin/env bash
# Validate paired exponential scoring on real candidates, then measure conic deadline flips under load.
set -uo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
export PATH="$HOME/.local/bin:$PATH"
store=${STORE:-store-iso}
out=$store/experiment
note() { echo "[$(date '+%F %T')] $*" | tee -a "$out/timeline.log"; }
while pgrep -f '[a]rena.cli --store' >/dev/null; do sleep 20; done

candidates=("$@")
for tar in "${candidates[@]}"; do
  note "quiet paired grade of $tar"
  ./bench --store "$store" interference --levels 1 --candidate "$tar" >> "$out/candidates-quiet.log" 2>&1
done
note "all candidates plus references at 4 concurrent graders, 10 development slots under load"
args=()
for tar in "${candidates[@]}"; do args+=(--candidate "$tar"); done
./bench --store "$store" interference --levels 4 --dev-load 10 "${args[@]}" > "$out/candidates-loaded.log" 2>&1

note "conic baseline alone"
./bench --store "$store" baseline --task conic > "$out/conic-baseline.log" 2>&1
note "conic references at 4 concurrent graders, 10 development slots under load"
./bench --store "$store" interference --task conic --levels 4 --dev-load 10 > "$out/conic-loaded.log" 2>&1
note "validation done"
