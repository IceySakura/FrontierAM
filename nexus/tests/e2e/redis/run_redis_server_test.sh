#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

python3 -c 'import redis, zmq' >/dev/null || {
  echo 'Redis and ZMQ dependencies are required: uv sync --extra redis' >&2
  exit 1
}
exec python3 -m tests.e2e.redis.run_cluster_tests suite "$@"
