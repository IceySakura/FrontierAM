#!/usr/bin/env bash
# Run attempts one at a time (for a model whose provider quota cannot serve two attempts), each fully graded
# before the next starts. Waits for any running campaign, then re-seals the runtime once.
#   scripts/rounds/serial-attempts.sh TASK:MODEL...
set -uo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
export PATH="$HOME/.local/bin:$PATH"
while pgrep -f '[s]cripts/campaign.sh' >/dev/null; do sleep 30; done
mv .runtime ".runtime.superseded-$(date +%s)"
./bench prepare --codex "$(command -v codex)"
for pair in "$@"; do
  ./scripts/rounds/campaign.sh "$pair"
done
