#!/bin/bash
# exponential-v2, private half. Run only after the panel is frozen on public data (scripts/exp2-public.sh, then the
# calibrated parameters committed), and only in an idle window: it loads every non-housekeeping CPU.
#   1. synthetic private mothers (sealed seeds) and the private standard operators (BFS-derived where too large);
#   2. private cases and their reference solutions;
#   3. re-seal .runtime-exp2 with the private cases graded and the public cases for development.
# Log: store/reruns/exp2-private.log
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
DATA=tasks/exponential/sota_v1/dataset_v2
OUT=tasks/exponential/sota_v1/calibration_v2
IMAGE=frontieram-exponential:port-v1
LOG=store/reruns/exp2-private.log
log() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> "$LOG"; }
container() {
    local cpus=$1 memory=$2; shift 2
    docker run --rm --network none --cpuset-cpus "$cpus" --cgroup-parent frontieram-dev.slice --memory "$memory" \
        -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -v "$ROOT:$ROOT" -w "$ROOT" --entrypoint /bin/sh "$IMAGE" -c "$*"
}
[ -f "$OUT/FROZEN.json" ] || { echo "panel not frozen: $OUT/FROZEN.json missing" >&2; exit 1; }
[ -f "$DATA/raw/DIMACS10__road_usa.tar.gz" ] || { echo "road_usa not downloaded yet" >&2; exit 1; }

log '"event": "start"'
(cd $DATA && taskset -c 0-3 nice "$ROOT/.venv-ops/bin/python" synth.py --split private)
(cd $DATA && taskset -c 0-3 nice "$ROOT/.venv-ops/bin/python" fetch_suitesparse.py --split private) || true
(cd $DATA && taskset -c 0-3 nice "$ROOT/.venv-ops/bin/python" build_cases.py --split private)
log "\"event\": \"private_cases_built\", \"cases\": $(python3 -c "import json;print(len(json.load(open('$DATA/cases/private/manifest.json'))['cases']))")"
# TRUTH_CPUS leaves CPUs to an attempt still developing (default: every non-housekeeping CPU).
TRUTH_CPUS=${TRUTH_CPUS:-4-31}
JOBS=$(python3 -c "import sys;print(sum(int(b)-int(a)+1 if '-' in r else 1 for r in sys.argv[1].split(',') for a,b in [(r.split('-')+[r])[:2]]))" "$TRUTH_CPUS")
container "$TRUTH_CPUS" 48g "cd $DATA && python3 truth.py cases/private/manifest.json --jobs $JOBS" > "$OUT/truth-private.log" 2>&1
log "\"event\": \"private_truth\", \"summary\": $(tail -1 "$OUT/truth-private.log")"
.venv-ops/bin/python scripts/prepare_exponential_v2.py --force --graded $DATA/cases/private \
    --public-dev $DATA/cases/public_smoke --public-full $DATA/cases/public_dev_full > "$OUT/seal-private.json"
log '"event": "sealed"'
