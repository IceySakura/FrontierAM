#!/usr/bin/env bash
set -euo pipefail
if [[ "$(cat /app/result.txt)" == "hello-from-sidecar" ]]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
