#!/bin/bash
# Disk maintenance for an idle window (no timed grading): delete regenerable data, then move /data/archive to the
# personal cloud disk. Each group is packed (tar + zstd), encrypted (gpg AES256, passphrase in /root/.frontieram),
# uploaded, confirmed synced by the cloud-disk client, read back and checked, and only then removed locally.
# Usage: scripts/ops/data-maintenance.sh   (log: var/stores/store/reruns/data-maintenance.log)
set -euo pipefail
cd "$(dirname "$0")/../.."
SPACE=01a147b3-5c57-7911-b246-48568aab4f1b
CLOUD=/agentdrive/Anydev/FrontierAM/archive/20261105
STAGE=/data/archive-upload
KEY=/root/.frontieram/archive-passphrase
LOG=var/stores/store/reruns/data-maintenance.log
QUIET=(taskset -c 0-3 nice -n 19 ionice -c3)
log() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> "$LOG"; }
synced() {
    curl -s -m 10 "http://127.0.0.1:49252/api/v1/spaces/$SPACE/sync/status" | python3 -c "
import json,sys; d=json.load(sys.stdin)['data']; sys.exit(0 if d['status']=='synced' and d['remain']==0 and not d['has_pending_operations'] else 1)"
}
wait_synced() {
    for _ in $(seq 1 720); do synced && return 0; sleep 10; done
    log '"event": "sync_timeout"'; return 1
}

# 1. Regenerable data
if [ -d /data/trash/20261105 ]; then
    "${QUIET[@]}" rm -rf /data/trash/20261105
    log '"event": "trash_deleted", "path": "/data/trash/20261105"'
fi
for image in local/four-angle-public:b1-v1.2 songberg-cpu-probe:r4.2.2-33d65cf 9eadc627ef71 627b89c51413; do
    docker image rm "$image" > /dev/null 2>&1 && log "\"event\": \"image_removed\", \"image\": \"$image\"" || true
done

# 2. Cloud throughput probe (1 GiB)
mkdir -p "$CLOUD" "$STAGE"
"${QUIET[@]}" head -c 1073741824 /dev/urandom > "$STAGE/probe.bin"
start=$(date +%s.%N)
"${QUIET[@]}" cp "$STAGE/probe.bin" "$CLOUD/probe.bin"
wait_synced
seconds=$(python3 -c "import time;print(round(time.time()-$start,1))")
cmp -s "$STAGE/probe.bin" "$CLOUD/probe.bin" && probe=match || probe=mismatch
rm -f "$CLOUD/probe.bin" "$STAGE/probe.bin"
log "\"event\": \"throughput_probe\", \"bytes\": 1073741824, \"seconds\": $seconds, \"readback\": \"$probe\""
[ "$probe" = match ]

# 3. Archive groups
cp /data/archive/README.md /data/archive/MOVES.tsv "$CLOUD/"
for group in FrontierAM/runtimes FrontierAM/stores FrontierAM/misc FrontierAM-orig; do
    [ -e "/data/archive/$group" ] || continue
    name=$(echo "$group" | tr / _)
    out="$STAGE/$name.tar.zst.gpg"
    "${QUIET[@]}" bash -c "tar -C /data/archive -cf - '$group' | zstd -q -T4 --long=31 -12 | \
        gpg --batch --yes --symmetric --cipher-algo AES256 --passphrase-file '$KEY' -o '$out'"
    sum=$(sha256sum "$out" | cut -d' ' -f1)
    listing=$(du -sb "/data/archive/$group" | cut -f1)
    "${QUIET[@]}" rsync -a "$out" "$CLOUD/"
    wait_synced
    remote=$(sha256sum "$CLOUD/$name.tar.zst.gpg" | cut -d' ' -f1)
    if [ "$remote" != "$sum" ]; then
        log "\"event\": \"verify_failed\", \"group\": \"$group\""; exit 1
    fi
    # Decrypt and list the uploaded copy end to end before deleting anything local.
    gpg --batch --quiet --decrypt --passphrase-file "$KEY" "$CLOUD/$name.tar.zst.gpg" | zstd -q -d --long=31 | tar -tf - > /dev/null
    echo "$sum  $name.tar.zst.gpg" >> "$CLOUD/SHA256SUMS"
    "${QUIET[@]}" rm -rf "/data/archive/$group" "$out"
    log "\"event\": \"archived\", \"group\": \"$group\", \"tar_bytes\": $listing, \"encrypted_bytes\": $(stat -c %s "$CLOUD/$name.tar.zst.gpg"), \"sha256\": \"$sum\""
done
rmdir "$STAGE" 2>/dev/null || true
log "\"event\": \"done\", \"data_free\": \"$(df -h /data | awk 'NR==2{print $4}')\""
