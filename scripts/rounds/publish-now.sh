#!/usr/bin/env bash
# Publish every v6.3 cell that has a score, rebuild the task pages and the dashboard, commit and push to git.woa and
# restart the dashboard service; sync the GitHub mirror at most every 30 minutes. Called by the grade loops after
# each graded cell and by ihvp-v4-finals.sh after each private final; concurrent calls are serialized.
# Log: var/stores/store/reruns/publish.log
set -uo pipefail
cd "$(dirname "$0")/../.."
exec 9> var/stores/store/reruns/publish.lock
flock 9
{
    echo "{\"time\": \"$(date +%FT%T)\", \"event\": \"publish\", \"why\": \"${1:-}\"}"
    nexus/.venv/bin/python scripts/analyze/ihvp-v4-runs.py scores
    python3 scripts/rounds/v63-publish.py
    python3 scripts/site/build_task_pages.py > /dev/null
    .venv-ops/bin/python scripts/site/build_dashboard.py > /dev/null
    git add README.md dashboard tasks/README.md tasks/*/README.md tasks/*/meta.json tasks/*/solutions
    if ! git diff --cached --quiet; then
        git commit -q -m "auto-publish: ${1:-graded cells}" && timeout 300 git push -q origin HEAD
        systemctl restart frontieram-dashboard-build.service
    fi
    stamp=var/stores/store/reruns/github-sync.stamp
    if [ ! -f $stamp ] || [ $(( $(date +%s) - $(stat -c %Y $stamp) )) -gt 1800 ]; then
        timeout 900 scripts/ops/sync_github.sh && touch $stamp
    fi
} >> var/stores/store/reruns/publish.log 2>&1
