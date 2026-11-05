#!/bin/bash
# Build the amd64 task images of every operator task (sealing them again with the image IDs recorded), then the v6
# development harness image on top of each, the same way as for exponential and conic.
# Usage: scripts/build-operator-images.sh [--no-harness]
set -euo pipefail
cd "$(dirname "$0")/.."
ATR=${ATR:-/data/workspace/agent-traffic-runtime}
python3 scripts/prepare_operator_tasks.py --build-images --force
for spec in tasks/operator/*/task.json; do
    name=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['name'])" "$spec")
    tag=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['images']['task']['tag'])" "$spec")
    echo "== $name: $tag"
    docker run --rm --network none --entrypoint sh "$tag" -c \
        'uname -m; python3 -c "import sys,numpy; print(sys.version.split()[0], numpy.__version__)"; ls /workspace'
    if [ "${1:-}" != "--no-harness" ]; then
        docker buildx build --load --build-context atr="$ATR" --build-arg BASE="$tag" \
            -t "frontieram-$name-harness:v6" -f docker/Dockerfile.harness docker
    fi
done
