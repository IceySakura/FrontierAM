#!/bin/bash
# Copy the sealed operator-task images into every grading daemon by ID. Each daemon runs on its slot's SMT sibling,
# so a slot is loaded only while it holds no broker lease and runs no container; busy slots are retried until free.
# Usage: scripts/setup/load-grade-images.sh [slot ...]   (default: every grade slot in /etc/frontieram/machine.json)
set -euo pipefail
cd "$(dirname "$0")/../.."
slots=("$@")
if [ ${#slots[@]} -eq 0 ]; then
    mapfile -t slots < <(python3 -c "import json;[print(s['id']) for s in json.load(open('/etc/frontieram/machine.json'))['slots'] if s['kind']=='grade']")
fi
mapfile -t images < <(python3 -c "
import json,glob
for p in sorted(glob.glob('.runtime-ops/*/lock.json')):
    image = json.load(open(p))['image']
    if image.get('id'): print(image['tag'], image['id'])")
pending=("${slots[@]}")
while [ ${#pending[@]} -gt 0 ]; do
    next=()
    for slot in "${pending[@]}"; do
        host=unix:///run/frontieram-grade/$slot/docker.sock
        # A grade between two sessions has no container but still holds its lease (quiet grades lease every slot).
        owner=$(python3 -c "import json,sys
try: print(json.load(open('/run/frontieram/slots/$slot.json')).get('owner') or '')
except (OSError, ValueError): print('')")
        if [ -n "$owner" ] || [ "$(docker --host "$host" ps -q | wc -l)" != 0 ]; then
            next+=("$slot")
            continue
        fi
        for entry in "${images[@]}"; do
            read -r tag id <<< "$entry"
            if ! docker --host "$host" image inspect "$id" > /dev/null 2>&1; then
                docker save "$tag" | docker --host "$host" image load --quiet
            fi
            [ "$(docker --host "$host" image inspect --format '{{.Id}}' "$id")" = "$id" ]
        done
        echo "$(date +%T) $slot loaded ${#images[@]} images"
    done
    pending=("${next[@]}")
    [ ${#pending[@]} -gt 0 ] && sleep 60
done
