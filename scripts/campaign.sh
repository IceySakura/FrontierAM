#!/usr/bin/env bash
# Development attempts plus grading under the measured policies, logged to $STORE/campaign-<stamp>/.
#   scripts/campaign.sh [--rebaseline] [--regrade SUBMISSION]... TASK:MODEL...
set -uo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
export PATH="$HOME/.local/bin:$PATH"
KEY=${AIHUB_KEY_FILE:-/root/.frontieram/aihub.key}
store=${STORE:-store}
log=$store/campaign-$(date +%Y%m%dT%H%M%S)
mkdir -p "$log"
note() { echo "[$(date '+%F %T')] $*" | tee -a "$log/timeline.log"; }
queue_open() {
  ./bench --store "$store" status | python3 -c 'import json,sys; q=json.load(sys.stdin)["queue"]; print(q["pending"]+q["running"])'
}
rebaseline=0
regrades=()
attempts=()
while [ $# -gt 0 ]; do
  case "$1" in
    --rebaseline) rebaseline=1; shift ;;
    --regrade) regrades+=("$2"); shift 2 ;;
    *) attempts+=("$1"); shift ;;
  esac
done

note "campaign start: ${#attempts[@]} attempts, ${#regrades[@]} regrades, rebaseline=$rebaseline"
./bench --store "$store" host status > "$log/host.json" 2>&1

if [ "$rebaseline" = 1 ]; then
  ./bench --store "$store" baseline --task exponential --replace > "$log/baseline-exponential.log" 2>&1
  note "exponential baseline: $(grep -o '"qualified": [a-z]*' "$log/baseline-exponential.log" | tail -1)"
  ./bench --store "$store" baseline --task conic --replace > "$log/baseline-conic.log" 2>&1
  note "conic baseline: $(grep -o '"qualified": [a-z]*' "$log/baseline-conic.log" | tail -1)"
fi

./bench --store "$store" grade-worker --task exponential --workers 2 > "$log/worker-exponential.log" 2>&1 &
wexp=$!
./bench --store "$store" grade-worker --task conic --workers 2 > "$log/worker-conic.log" 2>&1 &
wcon=$!
for submission in "${regrades[@]}"; do
  python3 -c "import sys; sys.path.insert(0, '.'); from arena.store import Store; print(Store('$store').enqueue('$submission', note='regrade under final stack'))" \
    >> "$log/regrades.log"
  note "regrade queued: $submission"
done

devs=()
for pair in "${attempts[@]}"; do
  task=${pair%%:*}
  model=${pair#*:}
  ./bench --store "$store" develop --task "$task" --model "$model" --key-file "$KEY" \
    > "$log/develop-$task-$model.log" 2>&1 &
  devs+=($!)
  note "development started: $task / $model"
  sleep 5
done
[ ${#devs[@]} -gt 0 ] && wait "${devs[@]}"
note "development finished"

while [ "$(queue_open)" != 0 ]; do sleep 60; done
kill "$wexp" "$wcon" 2>/dev/null
wait "$wexp" "$wcon" 2>/dev/null
note "grading queue drained"
./bench --store "$store" sentinel --task exponential --count 1 > "$log/sentinel.log" 2>&1
./bench --store "$store" report > "$log/REPORT.md" 2>&1
note "report: $log/REPORT.md"
