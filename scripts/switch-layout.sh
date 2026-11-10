#!/bin/bash
# Change the grading layout once the machine is idle: wait for running grades and development to finish,
# re-run host setup, keep the old sealed runtime under a new name and prepare a fresh one for the new daemons.
# Usage: scripts/switch-layout.sh packed|spread
set -euo pipefail
cd "$(dirname "$0")/.."
placement=$1

idle() {
    [ -z "$(ls var/stores/store/jobs/running 2>/dev/null)" ] && ! pgrep -f "arena.cli.* develop " >/dev/null \
        && [ -z "$(ls /run/frontieram/slots/*.json 2>/dev/null)" ]
}
until idle; do
    sleep 30
done
echo "$(date -Is) idle; switching to $placement"
old=".runtime-$(python3 -c 'import json; print(json.load(open("/etc/frontieram/machine.json"))["grade_placement"])')-retired-$(date +%Y%m%dT%H%M%S)"
./bench host setup --grade-placement "$placement"
mv .runtime "$old"
echo "previous runtime kept as $old"
./bench prepare --codex "$(command -v codex)"
./bench host status
echo "$(date -Is) layout $placement ready"
