#!/usr/bin/env bash
# ihvp-sequence-v4: grade each new-model attempt on private once its submission is frozen (attempt ended), never
# earlier, then rebuild tasks/ihvp-sequence/analysis/V4_RESULTS.md. Log: var/stores/store/reruns/ihvp-v4-finals.log
set -uo pipefail
cd "$(dirname "$0")/../.."
export FRONTIERAM_RUNTIME=$PWD/.runtime-recal
frozen() { python3 -c "import json,sys;r=[json.loads(l) for l in open(sys.argv[1]) if 'develop_finished' in l];print(r[-1]['submission'] if r and r[-1]['status']=='frozen' else ('-' if r else ''))" "$1" 2>/dev/null; }
grade_one() {
    local tag=$1 setting=$2 sub
    until sub=$(frozen "var/stores/store/reruns/v63-$tag-$setting-ihvp-sequence-v4-1.log") && [ -n "$sub" ]; do sleep 60; done
    [ "$sub" = - ] && { echo "{\"label\": \"$tag-$setting\", \"submission\": null}"; return; }
    echo "{\"time\": \"$(date +%FT%T)\", \"event\": \"final_started\", \"label\": \"$tag-$setting\", \"submission\": \"$sub\"}"
    nexus/.venv/bin/python scripts/analyze/ihvp-v4-runs.py final "$sub" "$tag-$setting"
    scripts/rounds/publish-now.sh "ihvp v4 $tag-$setting"
}
for tag in ${V4_TAGS:-sol opus}; do for setting in single iterative; do grade_one $tag $setting & done; done
wait
nexus/.venv/bin/python scripts/analyze/ihvp-v4-runs.py report
echo "{\"time\": \"$(date +%FT%T)\", \"event\": \"finals_done\"}"
