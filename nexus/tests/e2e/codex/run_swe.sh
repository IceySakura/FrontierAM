#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NEXUS_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
GENERATOR="${SWE_GENERATOR:-$SCRIPT_DIR/swe_generator.py}"
UV_BIN="${UV_BIN:-uv}"
PYTHON_BIN="${PYTHON_BIN:-python}"

route="${CODEX_E2E_ROUTE:-legacy}"
case "$route" in
  legacy)
    checker="$SCRIPT_DIR/check_legacy_trace.py"
    use_gateway=false
    route_label="legacy/default path"
    ;;
  gateway)
    checker="$SCRIPT_DIR/check_gateway_trace.py"
    use_gateway=true
    route_label="general Gateway policy"
    ;;
  *)
    echo "CODEX_E2E_ROUTE must be legacy or gateway: $route" >&2
    exit 2
    ;;
esac

export GONGFENG_RUNTIME_TOKEN="${GONGFENG_RUNTIME_TOKEN:-${GONGFENG_TOKEN:-${GF_TOKEN:-}}}"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY to APP_ID:APP_KEY}"
: "${GONGFENG_RUNTIME_TOKEN:?set GONGFENG_RUNTIME_TOKEN (or GONGFENG_TOKEN)}"
: "${SWE_INSTANCE_IDS:?set a comma-separated fixed SWE sample}"
: "${SWE_TRACE_OUTPUT_DIR:?set the parent output directory}"

export CODEX_VERSION="${CODEX_VERSION:-0.114.0}"
export CODEX_REQUIRE_COMPACTION="${CODEX_REQUIRE_COMPACTION:-false}"
export CODEX_REQUIRE_MULTI_AGENT="${CODEX_REQUIRE_MULTI_AGENT:-false}"
for boolean_name in CODEX_REQUIRE_COMPACTION CODEX_REQUIRE_MULTI_AGENT; do
  boolean_value="${!boolean_name}"
  if [[ "$boolean_value" != "true" && "$boolean_value" != "false" ]]; then
    echo "$boolean_name must be true or false" >&2
    exit 2
  fi
done

version_minor="$(cut -d. -f2 <<<"$CODEX_VERSION")"
if ! [[ "$version_minor" =~ ^[0-9]+$ ]]; then
  echo "CODEX_VERSION must have a numeric minor component: $CODEX_VERSION" >&2
  exit 2
fi
if [[ "$CODEX_REQUIRE_COMPACTION" == "true" ]]; then
  export CODEX_MODEL_CONTEXT_WINDOW="${CODEX_MODEL_CONTEXT_WINDOW:-32768}"
  export CODEX_AUTO_COMPACT_TOKEN_LIMIT="${CODEX_AUTO_COMPACT_TOKEN_LIMIT:-16384}"
  if (( version_minor >= 146 )); then
    export CODEX_AUTO_COMPACT_TOKEN_LIMIT_SCOPE="${CODEX_AUTO_COMPACT_TOKEN_LIMIT_SCOPE:-total}"
  else
    unset CODEX_AUTO_COMPACT_TOKEN_LIMIT_SCOPE
  fi
  export CODEX_COMPACT_PROMPT="${CODEX_COMPACT_PROMPT:-You are performing a CONTEXT CHECKPOINT COMPACTION. Summarize the work so another agent can continue without losing decisions, edits, test results, or remaining tasks.}"
else
  unset CODEX_MODEL_CONTEXT_WINDOW
  unset CODEX_REASONING_EFFORT
  unset CODEX_AUTO_COMPACT_TOKEN_LIMIT
  unset CODEX_AUTO_COMPACT_TOKEN_LIMIT_SCOPE
  unset CODEX_COMPACT_PROMPT
fi

if [[ "$CODEX_REQUIRE_MULTI_AGENT" == "true" ]]; then
  if (( version_minor < 146 )); then
    echo "forced multi-agent validation requires Codex 0.146 or newer" >&2
    exit 2
  fi
  if [[ -z "${CODEX_EXTRA_CLI_ARGS_JSON:-}" ]]; then
    export CODEX_EXTRA_CLI_ARGS_JSON='["-c","features.multi_agent=true"]'
  fi
  if [[ -z "${CODEX_DEVELOPER_INSTRUCTIONS:-}" ]]; then
    export CODEX_DEVELOPER_INSTRUCTIONS="Before editing, delegate exactly one bounded read-only investigation of the reported bug to one subagent with spawn_agent, wait for that child to finish, use its result, and do not spawn any additional subagents. Then implement and verify the fix yourself."
  fi
fi

if [[ -z "${SWE_DATASET:-}" && -z "${SWE_DATASET_PATH:-}" ]]; then
  export SWE_DATASET=swe-public/SWE-bench_Verified
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/hytuner_codex_uv_cache}"
export SWE_AGENT_NAME=codex
export SWE_WORKER_VENV=1
export SWE_RECORD_TRAJECTORY=1
export SWE_MAX_CONCURRENT=1
export SWE_RAISE_ON_RUNTIME_ERROR=1
export SWE_RUNTIME_TIMEOUT="${SWE_RUNTIME_TIMEOUT:-3600}"
export CODEX_TIMEOUT="${CODEX_TIMEOUT:-3600}"
export CODEX_MODEL="${SWE_MODEL:-${CODEX_MODEL:-api_deepseek_deepseek-v4-flash}}"
export OPENAI_CLIENT_NAME=openai
export OPENAI_MODEL="$CODEX_MODEL"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${BASE_URL:-http://llm-api.model-eval.woa.com/v1}}"
export SWE_USE_GATEWAY="$use_gateway"

artifact_dir="$SWE_TRACE_OUTPUT_DIR/$CODEX_VERSION"
database="$artifact_dir/$route.db"
results="$artifact_dir/$route.jsonl"
mkdir -p "$artifact_dir"

echo "Running Codex $CODEX_VERSION through the $route_label from $NEXUS_ROOT"
echo "SWE instances: $SWE_INSTANCE_IDS"
echo "Forced compaction: $CODEX_REQUIRE_COMPACTION"
echo "Required single subagent: $CODEX_REQUIRE_MULTI_AGENT"
echo "Artifacts: $database, $results"
(
  cd "$NEXUS_ROOT"
  "$UV_BIN" run --with dataclasses-json --with unidiff --with filelock nexus run-batch \
    --generator "$GENERATOR" \
    --database "$database" \
    --max-concurrent 1 \
    --timeout "$((SWE_RUNTIME_TIMEOUT + 900))" \
    --export-format jsonl \
    --export-path "$results"
)

checker_args=("$results" "$database")
if [[ "$route" == "gateway" ]]; then
  checker_args+=(--expected-version "$CODEX_VERSION")
fi
if [[ "$CODEX_REQUIRE_COMPACTION" == "true" ]]; then
  checker_args+=(--require-compaction)
fi
if [[ "$CODEX_REQUIRE_MULTI_AGENT" == "true" ]]; then
  checker_args+=(--require-multi-agent)
fi
PYTHONPATH="$NEXUS_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$checker" "${checker_args[@]}"
