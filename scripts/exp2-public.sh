#!/bin/bash
# exponential-v2, public half (idle window only: it loads every non-housekeeping CPU).
#   1. convert the public standard matrices (PSD check with shift-invert fallback) and build the full-size public cases;
#   2. reference solutions (truth.py) in one container over CPUs 4-31;
#   3. panel calibration on the public cases under each member's frozen rule (expokit, expmv), and validation of the
#      members without a parameter (kiops, scipy, chebyshev), each in its own single-CPU container;
#   4. the one_shot subset for the development package (cases/public_dev_full).
# Private cases are generated only after the panel is frozen (scripts/exp2-private.sh).
# Log: store/reruns/exp2-public.log; reports under tasks/exponential/sota_v1/calibration_v2/.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
V2=tasks/exponential/sota_v1
DATA=$V2/dataset_v2
OUT=$V2/calibration_v2
IMAGE=frontieram-exponential:port-v1
LOG=store/reruns/exp2-public.log
log() { echo "{\"time\": \"$(date +%FT%T)\", $1}" >> "$LOG"; }
container() {  # container CPUSET MEMORY COMMAND...  (development slice, offline, repository mounted at the same path)
    local cpus=$1 memory=$2; shift 2
    docker run --rm --network none --cpuset-cpus "$cpus" --cgroup-parent frontieram-dev.slice --memory "$memory" \
        -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -v "$ROOT:$ROOT" -w "$ROOT" --entrypoint /bin/sh "$IMAGE" -c "$*"
}
mkdir -p "$OUT"

log '"event": "start"'
(cd $DATA && taskset -c 0-3 nice "$ROOT/.venv-ops/bin/python" fetch_suitesparse.py --split public) || true
(cd $DATA && taskset -c 0-3 nice "$ROOT/.venv-ops/bin/python" build_cases.py --split public)
log "\"event\": \"public_cases_built\", \"cases\": $(python3 -c "import json;print(len(json.load(open('$DATA/cases/public/manifest.json'))['cases']))")"

container 4-31 48g "cd $DATA && python3 truth.py cases/public/manifest.json --jobs 28" > "$OUT/truth-public.log" 2>&1
log "\"event\": \"public_truth\", \"summary\": $(tail -1 "$OUT/truth-public.log")"

MANIFEST=$ROOT/$DATA/cases/public/manifest.json
TOOLS=$ROOT/tasks/exponential/public/tools
P=$ROOT/$V2/panel
container 8 6g "python3 $P/expokit/calibrate.py $TOOLS $MANIFEST $P/expokit/expokit_avx2 $ROOT/$OUT/expokit.json" > "$OUT/expokit.log" 2>&1 &
container 10 6g "python3 $P/expmv/calibrate.py $TOOLS $MANIFEST $ROOT/$OUT/expmv.json" > "$OUT/expmv.log" 2>&1 &
for member in kiops scipy chebyshev; do
    cpu=$((12 + 2 * $(echo "kiops scipy chebyshev" | tr ' ' '\n' | grep -n "^$member$" | cut -d: -f1)))
    container $cpu 6g "python3 $TOOLS/run_local.py --manifest $MANIFEST --command $P/$member/run.sh --repeats 1 \
        --output $ROOT/$OUT/$member-validation.json" > "$OUT/$member.log" 2>&1 &
done
wait
log '"event": "panel_calibration_done"'

python3 - "$DATA" <<'EOF'
import json, shutil, sys
from pathlib import Path
root = Path(sys.argv[1]) / 'cases'
manifest = json.loads((root / 'public/manifest.json').read_text())
target = root / 'public_dev_full'
shutil.rmtree(target, ignore_errors=True)
target.mkdir()
keep = [c for c in manifest['cases'] if c['scenario'] == 'one_shot']
for case in keep:
    shutil.copyfile(root / 'public' / case['file'], target / case['file'])
(target / 'manifest.json').write_text(json.dumps({**manifest, 'scope': 'public dataset-v2, one_shot subset for development',
                                                  'cases': keep}, indent=2) + '\n')
print(len(keep))
EOF
log '"event": "done"'
