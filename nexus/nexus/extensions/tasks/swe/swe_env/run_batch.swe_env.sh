#!/usr/bin/env bash
# Example: evaluate MANY SWE-Env instances in parallel from an instances.jsonl.
#
#   bash nexus/extensions/tasks/swe/swe_env/run_batch.swe_env.sh /data/swe-env/instances.jsonl
#
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

# --- Data source -------------------------------------------------------------
export SWE_DATASET_PATH="${1:-${SWE_DATASET_PATH:-/path/to/swe-env/instances.jsonl}}"
# Optional subset / paging:
export SWE_INSTANCE_IDS="${SWE_INSTANCE_IDS:-}"
# export SWE_SKIP=0
# export SWE_LIMIT=0   # unset/0 = no limit

# --- Sandbox (Gongfeng) ------------------------------------------------------
export SWE_RUNTIME_PROVIDER_NAME="${SWE_RUNTIME_PROVIDER_NAME:-gongfeng}"
# Sandbox API base URL: prefer SWE_RUNTIME_BASE_URL, fall back to the legacy GONGFENG_API_BASE_URL.
export SWE_RUNTIME_BASE_URL="${SWE_RUNTIME_BASE_URL:-${GONGFENG_API_BASE_URL:-http://hyrl-sandbox.prod.woa.com:8052}}"
# Sandbox API token: prefer GONGFENG_RUNTIME_TOKEN, fall back to the legacy GONGFENG_TOKEN.
export GONGFENG_RUNTIME_TOKEN="${GONGFENG_RUNTIME_TOKEN:-${GONGFENG_TOKEN:-?set GONGFENG_RUNTIME_TOKEN (or legacy GONGFENG_TOKEN) (sandbox API token)}}"
export SWE_RUNTIME_TIMEOUT="${SWE_RUNTIME_TIMEOUT:-7200}"
export SWE_RUNTIME_CPU="${SWE_RUNTIME_CPU:-2}"
export SWE_RUNTIME_MEMORY="${SWE_RUNTIME_MEMORY:-10}"

# --- Agent + host-side LLM ---------------------------------------------------
export SWE_AGENT_NAME="${SWE_AGENT_NAME:-swe_agent}"
export SWE_MAX_ITERATIONS="${SWE_MAX_ITERATIONS:-100}"
export OPENAI_CLIENT_NAME="${OPENAI_CLIENT_NAME:-litellm}"
export OPENAI_MODEL="${OPENAI_MODEL:-bedrock/anthropic.claude-opus-4-8}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://trpc-gpt-eval.production.polaris:8080}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY}"

# Upload the current nexus wheel into the eval sandbox so the swe-env dataset handler is importable there (the runtime image may not have it baked in).
export SWE_WORKER_VENV="${SWE_WORKER_VENV:-1}"
# Re-raise infra errors so failed instances can be retried by the batch layer.
export SWE_RAISE_ON_RUNTIME_ERROR="${SWE_RAISE_ON_RUNTIME_ERROR:-1}"
# Whether to run the oracle (only for debugging)
export SWE_RUN_ORACLE="${SWE_RUN_ORACLE:-0}"
# Whether to record the agent's LLM trajectories (messages + per-branch tools) via the chat_tracker processor
export SWE_RECORD_TRAJECTORY="${SWE_RECORD_TRAJECTORY:-1}"
# Tools mode for chat_tracker.
export SWE_TOOLS_MODE="${SWE_TOOLS_MODE:-branch_last}"

MAX_CONCURRENT="${MAX_CONCURRENT:-20}"
GENERATOR_PATH="${GENERATOR_PATH:-nexus/extensions/tasks/swe/generator.py}"
DB="${DB:-./outputs/swe_env_eval_batch.db}"
EXPORT_PATH="${EXPORT_PATH:-./outputs/swe_env_eval/batch_results.jsonl}"
mkdir -p "$(dirname "${DB}")" "$(dirname "${EXPORT_PATH}")"

uv run nexus run-batch \
    -g "${GENERATOR_PATH}" \
    -d "${DB}" \
    --max-concurrent "${MAX_CONCURRENT}" \
    --retry-failed 1 \
    --export-format jsonl \
    --export-path "${EXPORT_PATH}"

echo "Batch DB:        ${DB}"
echo "Results (jsonl): ${EXPORT_PATH}"
