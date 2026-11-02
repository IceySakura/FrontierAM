#!/usr/bin/env bash
# SWE-Cosmos batch via run-part YAML (configs/{runtime,agents,llms})
#
#   bash nexus/extensions/tasks/swe/swe_cosmos/run_batch.swecosmos.sh /data/swecosmos/instances.jsonl
#
# 必填环境变量：GONGFENG_RUNTIME_TOKEN（或 GONGFENG_TOKEN）
# 选哪个 config: RUNTIME_CONFIG / AGENT_CONFIG / LLM_CONFIG（stem 或路径）
# 数据集相关：INCLUDE_INSTANCES / EXCLUDE_INSTANCES / SWE_SKIP / SWE_LIMIT
# 其余参数在三份 config YAML 的 ${oc.env:…} 里，见各文件头注释
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

export DATASET_PATH="${1:-${DATASET_PATH:-${SWE_DATASET_PATH:-}}}"
# export INCLUDE_INSTANCES="django__django-10097,/data/subset.jsonl"
# export EXCLUDE_INSTANCES="/data/swecosmos/skip.jsonl"
# export SWE_SKIP=0
# export SWE_LIMIT=1

export RUNTIME_CONFIG="${RUNTIME_CONFIG:-swecosmos_runtime}"
export AGENT_CONFIG="${AGENT_CONFIG:-claude_code}"
export LLM_CONFIG="${LLM_CONFIG:-claude-opus-5.litellm}"

export GONGFENG_RUNTIME_TOKEN="${GONGFENG_RUNTIME_TOKEN:-${GONGFENG_TOKEN:-}}"
: "${GONGFENG_RUNTIME_TOKEN:?set GONGFENG_RUNTIME_TOKEN (or GONGFENG_TOKEN)}"

export GONGFENG_PURPOSE="${GONGFENG_PURPOSE:-prod}"
export RUN_ORACLE="${RUN_ORACLE:-false}"

export MAX_CONCURRENT="${MAX_CONCURRENT:-20}"
export WORKER_PROCESSES="${WORKER_PROCESSES:-$(( (MAX_CONCURRENT + 24) / 25 ))}"
if [ "${WORKER_PROCESSES}" -gt 64 ]; then
    export WORKER_PROCESSES=64
fi
export OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
export DB="${DB:-${OUTPUT_DIR}/swecosmos_batch.db}"
export EXPORT_PATH="${EXPORT_PATH:-${OUTPUT_DIR}/swecosmos_batch.jsonl}"
mkdir -p "${OUTPUT_DIR}" "$(dirname "${DB}")" "$(dirname "${EXPORT_PATH}")"

uv run nexus run-batch \
    -g nexus/extensions/tasks/swe/swe_cosmos/generator.py \
    -d "${DB}" \
    --worker-processes "${WORKER_PROCESSES}" \
    --max-concurrent "${MAX_CONCURRENT}" \
    --retry-failed 1 \
    --export-format jsonl \
    --export-path "${EXPORT_PATH}"

echo "Batch DB:        ${DB}"
echo "Results (jsonl): ${EXPORT_PATH}"
