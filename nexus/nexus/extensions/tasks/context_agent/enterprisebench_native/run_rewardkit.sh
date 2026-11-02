#!/bin/bash
set -e
export LITELLM_DROP_PARAMS=true
# The sandbox has no public egress; do not fetch a pricing table during imports.
export LITELLM_LOCAL_MODEL_COST_MAP=True

# Keep RewardKit/OpenCode semantics identical to the source benchmark. The
# Nexus-native task owns judge.toml, opencode.json, and the tracing adapter.
python3 /tests/judge_runtime.py stage dependencies

mkdir -p /judge_workspace
python3 /tests/judge_runtime.py stage grading_context python3 /tests/fetch_grading_context.py /judge_workspace/external_services_diff.md

WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT
cp -a /tests/. "$WORK_DIR/"
rm -f "$WORK_DIR/fetch_grading_context.py"
export OPENCODE_CONFIG="$WORK_DIR/opencode.json"

python3 /tests/judge_runtime.py stage rewardkit python3 -P -c "
import os
import sys
import litellm
litellm.num_retries = 4
from rewardkit.__main__ import main
mcl = os.environ.get('REWARDKIT_MAX_CONCURRENT_LLM', '4')
mac = os.environ.get('REWARDKIT_MAX_CONCURRENT_AGENT', '2')
sys.argv = [
    'rewardkit', sys.argv[1], '--workspace', '/workdir',
    '--max-concurrent-llm', mcl, '--max-concurrent-agent', mac,
]
main()
" "$WORK_DIR"
