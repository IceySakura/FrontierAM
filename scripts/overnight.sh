#!/usr/bin/env bash
# Unattended campaign: fresh exponential anchor, conic baseline, five models x two tasks, grading under the
# measured isolation policies, a closing sentinel and a report. Every step logs to store/overnight/.
set -uo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
export PATH="$HOME/.local/bin:$PATH"
KEY=${AIHUB_KEY_FILE:-/root/.frontieram/aihub.key}
MODELS=(claude-opus-5-5 deepseek-v4-pro hy4-preview minimax-m3.1-flash step-5-preview)
log=store/overnight
mkdir -p "$log"
note() { echo "[$(date '+%F %T')] $*" | tee -a "$log/timeline.log"; }
queue_open() {
  ./bench status | python3 -c 'import json,sys; q=json.load(sys.stdin)["queue"]; print(q["pending"]+q["running"])'
}

note "campaign start (pid $$)"
while pgrep -f 'arena.cli interference' >/dev/null; do sleep 30; done
note "interference tests finished"

# Packed grade cores keep timed grading on its own L3 domain, away from development load.
./bench host setup --grade-placement packed > "$log/host-setup.json" 2>&1
note "host setup: $(python3 -c 'import json; print(json.load(open("'"$log"'/host-setup.json"))["ready"])')"

( while sleep 600; do
    echo "[$(date '+%F %T')] $(./bench host status | python3 -c 'import json,sys; d=json.load(sys.stdin); print("ready", d["ready"], [k for k, v in d["checks"].items() if not v])')"
  done ) >> "$log/host-health.log" 2>&1 &
health=$!

note "fresh exponential anchor (quiet, nothing else running)"
./bench baseline --task exponential --replace > "$log/baseline-exponential.log" 2>&1
note "exponential baseline: $(grep -o '"qualified": [a-z]*' "$log/baseline-exponential.log" | tail -1)"

./bench grade-worker --task exponential --workers 1 > "$log/worker-exponential.log" 2>&1 &
wexp=$!
./bench grade-worker --task conic --workers 1 > "$log/worker-conic.log" 2>&1 &
wcon=$!
./bench baseline --task conic > "$log/baseline-conic.log" 2>&1 &
note "grade workers started; conic baseline queued first"
sleep 20

devs=()
for task in exponential conic; do
  for model in "${MODELS[@]}"; do
    ./bench develop --task "$task" --model "$model" --key-file "$KEY" > "$log/develop-$task-$model.log" 2>&1 &
    devs+=($!)
    note "development queued: $task / $model (pid $!)"
    sleep 5
  done
done
wait "${devs[@]}"
note "all development attempts finished"

while [ "$(queue_open)" != 0 ]; do sleep 120; done
note "grading queue drained"

# Stop the looping workers first; otherwise one of them claims a sentinel and is killed mid-grade.
kill "$wexp" "$wcon" 2>/dev/null
wait "$wexp" "$wcon" 2>/dev/null
./bench sentinel --task exponential --count 2 > "$log/sentinel-exponential.log" 2>&1
note "closing exponential sentinels done"

kill "$health" 2>/dev/null
./bench rescore > "$log/rescore.log" 2>&1
./bench report > "$log/REPORT.md" 2>&1
note "report written to $log/REPORT.md"
