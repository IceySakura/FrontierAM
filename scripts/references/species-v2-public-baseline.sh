#!/usr/bin/env bash
# species-tree-v2 public-feedback baseline: both TREE-QMC references on the 50 shipped public cases, sharded on the
# grade slots; anchors for public-feedback scores in tasks/species-tree/analysis/species-tree-v2/public-anchors.json.
set -euo pipefail
cd "$(dirname "$0")/../.."
export FRONTIERAM_SPECIES_V2_DATA=public FRONTIERAM_RUNTIME=$PWD/.runtime-recal
suite=$(./bench --store var/stores/store-public-species-v2 suite --task species-tree-v2 --reference --shards 4 \
        --slots grade-06,grade-14,grade-22,grade-30 --isolation shared | tail -1 | python3 -c "import json,sys;print(json.load(sys.stdin)['id'])")
./bench --store var/stores/store-public-species-v2 grade-worker --task species-tree-v2 --workers 4 --drain
.venv-ops/bin/python scripts/grade/species-v2-suite-score.py --store var/stores/store-public-species-v2 --reference "$suite" \
    --out tasks/species-tree/references/r1/anchors-public.json
