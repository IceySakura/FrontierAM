#!/usr/bin/env bash
# SWE-Cosmos bugmaker batch via run-part YAML (configs/{runtime,llms})
#
#   bash nexus/extensions/tasks/swe_cosmos/run_batch.swecosmos_bugmaker.sh /data/swe-env/repos.jsonl
#
# 必填环境变量：GONGFENG_RUNTIME_TOKEN（或 GONGFENG_TOKEN）、LLM 的 API Key（见所选 llms/*.yaml 头注释）
# 选哪个 config: RUNTIME_CONFIG / LLM_CONFIG（configs/{runtime,llms}/ 下的 stem，或任意路径）
# 数据集相关：DATASET_PATH / INCLUDE_REPOS / EXCLUDE_REPOS / SWECOSMOS_SKIP / SWECOSMOS_LIMIT / REDO_EXISTING / SPLIT
# 其余参数在两份 config YAML 的 ${oc.env:…} 里，见各文件头注释
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

# .jsonl（每行一个 repo dict，repo_name 唯一）或一个装着 per-repo .json 的目录
export DATASET_PATH="${1:-${DATASET_PATH:-}}"
: "${DATASET_PATH:?set DATASET_PATH (a .jsonl of repo dicts, or a directory of per-repo .json)}"
# export INCLUDE_REPOS="swe-env.v1.0xMiden__miden-base-1955,/data/include.jsonl"
# export EXCLUDE_REPOS="/data/skip.json"
# 先跳过 N 个、再最多跑 N 个（都在过滤之后生效），不设 = 全跑。冒烟用 SWECOSMOS_LIMIT=1
# export SWECOSMOS_SKIP="0"
# export SWECOSMOS_LIMIT="1"
# 默认 false：已经跑完的 repo（result.json 判定）跳过，重跑即续跑
export REDO_EXISTING="${REDO_EXISTING:-false}"

export RUNTIME_CONFIG="${RUNTIME_CONFIG:-swecosmos_bugmaker_runtime}"
export LLM_CONFIG="${LLM_CONFIG:-claude-opus-5.openai}"

export GONGFENG_RUNTIME_TOKEN="${GONGFENG_RUNTIME_TOKEN:-${GONGFENG_TOKEN:-}}"
: "${GONGFENG_RUNTIME_TOKEN:?set GONGFENG_RUNTIME_TOKEN (or GONGFENG_TOKEN)}"

export GONGFENG_PURPOSE="${GONGFENG_PURPOSE:-prod}"

# 峰值 root 沙盒数。child 后端为 gongfeng 时，总沙盒数 ≈ MAX_CONCURRENT * (1 + SWECOSMOS_MAX_WORKERS)
export MAX_CONCURRENT="${MAX_CONCURRENT:-100}"
export WORKER_PROCESSES="${WORKER_PROCESSES:-$(( (MAX_CONCURRENT + 24) / 25 ))}"
if [ "${WORKER_PROCESSES}" -gt 64 ]; then
    export WORKER_PROCESSES=64
fi
export OUTPUT_DIR="${OUTPUT_DIR:-./outputs/swecosmos_bugmaker}"
export DB="${DB:-${OUTPUT_DIR}/swecosmos_bugmaker_batch.db}"
export EXPORT_PATH="${EXPORT_PATH:-${OUTPUT_DIR}/swecosmos_bugmaker_batch.jsonl}"
mkdir -p "${OUTPUT_DIR}" "$(dirname "${DB}")" "$(dirname "${EXPORT_PATH}")"

uv run nexus run-batch \
    -g nexus/extensions/tasks/swe_cosmos/generator.py \
    -d "${DB}" \
    --worker-processes "${WORKER_PROCESSES}" \
    --max-concurrent "${MAX_CONCURRENT}" \
    --retry-failed 1 \
    --export-format jsonl \
    --export-path "${EXPORT_PATH}"

echo "Batch DB:        ${DB}"
echo "Results (jsonl): ${EXPORT_PATH}"
echo "Per-repo output: ${OUTPUT_DIR}/<repo_name>/"
