#!/usr/bin/env bash
# The user simulator on the stock frontend_task. Edit the values below and run it.
# generator.py documents every variable.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -- dataset and container --
export FRONTEND_DATASET=/path/to/specifications.jsonl
export FRONTEND_IMAGE=mirrors.tencent.com/hunyuan_yanguan/webenv.claudecode.testset:v3
export FRONTEND_WORK_DIR=/workspace
export FRONTEND_TASK_LIMIT=1
export EVAL_TIMES=1

# -- worker skill, off unless a directory is given --
export FRONTEND_WORKER_SKILL=
export FRONTEND_WORKER_SKILL_DIRS=

# -- the user simulator --
export HITL_USER_PERSONA=default
export HITL_MAX_TURNS=350
export HITL_USER_CONTEXT_WINDOW=200000
export HITL_USER_MCP_SERVERS=

# -- the worker --
export HITL_WORKER_BACKEND=print_claude_code
export HITL_WORKER_CLIENT_PROFILE=default
export HITL_WORKER_CC_VERSION=2.1.117
export HITL_CC_TURN_TIMEOUT=14400
export HITL_WORKER_MCP_SERVERS=
export HITL_WORKER_MCP_SEARCH=0

# -- the run's ceiling: above a turn's, below the sandbox lifetime --
export TIMEOUT_MULTIPLIER=900

# -- models --
export OPENAI_API_KEY=
export OPENAI_BASE_URL=
export OPENAI_MODEL=
export HITL_HUMAN_LLM_TIMEOUT=600

# -- sandbox --
export SWE_RUNTIME_PROVIDER_NAME=gongfeng
export SWE_RUNTIME_BASE_URL=http://hyrl-sandbox.prod.woa.com:8052
export SWE_RUNTIME_TOKEN=
export SWE_RUNTIME_TIMEOUT=32400
# Sandbox lifetime, a Go duration up to 168h. Has to outlast the run's ceiling.
export SWE_RUNTIME_TTL=16h
export SWE_WORKER_VENV=true

CONCURRENT=1
OUTPUT_DIR="$PWD/hitl_frontend_runs"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DB="$OUTPUT_DIR/frontend_$(date +%Y%m%d_%H%M%S).db"

echo "dataset: $FRONTEND_DATASET"
echo "image:   $FRONTEND_IMAGE"
echo "persona: $HITL_USER_PERSONA"
echo "db:      $OUTPUT_DB"
echo

nexus run-batch \
    -g "$HERE/generator.py" \
    -d "$OUTPUT_DB" \
    --max-concurrent "$CONCURRENT" 2>&1 | tee "$OUTPUT_DB.log"

echo
echo "done: $OUTPUT_DB"
