#!/usr/bin/env bash
# Mirror this repository to GitHub without the private evaluation sets (user decision 2026-11-10: private test data
# lives only on git.woa.com). A fresh bare clone has the private paths removed from every commit with git-filter-repo,
# is checked for leftovers, and is force-pushed to the `github` remote. Filtering is deterministic, so unchanged history
# keeps the same commit IDs from one sync to the next.
# Excluded everywhere in history: */tests/private/*, */data/private/*, */cases/private/*, */docker_eval/private_pilot/*,
# and the private seed and manifest files of exponential v2.
# Usage: scripts/ops/sync_github.sh [branch ...]   (default: protocol-2h main)
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD
REMOTE=git@github-frontieram:IceySakura/FrontierAM.git
BRANCHES=("${@:-protocol-2h main}")
[ $# -eq 0 ] && BRANCHES=(protocol-2h main)
git show-ref --verify --quiet refs/heads/main || git branch main origin/main
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
git clone -q --no-local --bare "$REPO" "$work/mirror.git"
cd "$work/mirror.git"
"$REPO/.venv-ops/bin/git-filter-repo" --force --quiet --invert-paths \
    --path-glob '*/tests/private/*' --path-glob '*/data/private/*' --path-glob '*/cases/private/*' \
    --path-glob '*/docker_eval/private_pilot/*' --path-glob '*private_seeds.json' \
    --path-glob '*synthetic_private_manifest.json' --path-glob '*suitesparse_private_provenance.json'
left=$(git log --all --name-only --format= | grep -E '/tests/private/|/data/private/|/cases/private/|private_pilot/|private_seeds\.json|private_manifest|private_provenance' | sort -u | head -5 || true)
[ -z "$left" ] || { echo "private paths still present: $left" >&2; exit 1; }
git push -q --force --no-verify "$REMOTE" "${BRANCHES[@]}"
echo "pushed ${BRANCHES[*]} to GitHub without private evaluation data"
