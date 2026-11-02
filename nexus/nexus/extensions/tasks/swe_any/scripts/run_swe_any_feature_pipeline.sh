#!/usr/bin/env bash
# Additive SWE-Any feature-level pipeline:
#
#   repo preparation -> one batch item per feature -> deterministic collection
#
# This does not replace run_swe_any.sh or its six stage-specific configs.
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

if [[ -n "${1:-}" ]]; then
  export SWE_ANY_INSTANCES="$1"
fi
export SWE_ANY_INSTANCES="${SWE_ANY_INSTANCES:?set SWE_ANY_INSTANCES (or pass the .jsonl as \$1)}"
export SWE_ANY_OUTPUT_DIR="${SWE_ANY_OUTPUT_DIR:-./outputs/swe_any}"
export GONGFENG_TOKEN="${GONGFENG_TOKEN:?set GONGFENG_TOKEN (sandbox API token)}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY (host LLM for feature tasks)}"

MAX_CONCURRENT="${MAX_CONCURRENT:-20}"
PRE_MAX_CONCURRENT="${PRE_MAX_CONCURRENT:-2}"

PRE_DB="${PRE_DB:-${SWE_ANY_OUTPUT_DIR%/}/pipeline_pre_feature_batch.db}"
FEATURE_DB="${FEATURE_DB:-${SWE_ANY_OUTPUT_DIR%/}/pipeline_feature_batch.db}"
POST_DB="${POST_DB:-${SWE_ANY_OUTPUT_DIR%/}/pipeline_post_feature_batch.db}"

export SWE_ANY_PRE_FEATURE_RESULTS="${SWE_ANY_PRE_FEATURE_RESULTS:-${SWE_ANY_OUTPUT_DIR%/}/pipeline_pre_feature_batch_results.jsonl}"
export SWE_ANY_FEATURE_RESULTS="${SWE_ANY_FEATURE_RESULTS:-${SWE_ANY_OUTPUT_DIR%/}/pipeline_feature_batch_results.jsonl}"
POST_RESULTS="${POST_RESULTS:-${SWE_ANY_OUTPUT_DIR%/}/pipeline_post_feature_batch_results.jsonl}"

mkdir -p "${SWE_ANY_OUTPUT_DIR}"

echo "==================== swe_any feature pipeline: pre ===================="
if ! uv run nexus run-batch \
  -g nexus/extensions/tasks/swe_any/generator/pipeline/pre_feature.py \
  -d "${PRE_DB}" \
  --max-concurrent "${PRE_MAX_CONCURRENT}" \
  --worker-processes 1 \
  --retry-failed 1 \
  --export-format jsonl \
  --export-path "${SWE_ANY_PRE_FEATURE_RESULTS}"; then
  echo "WARN: pre-feature batch had partial failures; expanding successful manifests" >&2
fi

echo "==================== swe_any feature pipeline: features ===================="
# MAX_CONCURRENT is the only feature-level concurrency control.  A single
# worker process owns the DB and each task has one runtime with max_workers=1.
if ! uv run nexus run-batch \
  -g nexus/extensions/tasks/swe_any/generator/pipeline/feature.py \
  -d "${FEATURE_DB}" \
  --max-concurrent "${MAX_CONCURRENT}" \
  --worker-processes 1 \
  --retry-failed 1 \
  --export-format jsonl \
  --export-path "${SWE_ANY_FEATURE_RESULTS}"; then
  echo "WARN: feature batch had partial failures; post-feature will record them" >&2
fi

echo "==================== swe_any feature pipeline: post ===================="
uv run nexus run-batch \
  -g nexus/extensions/tasks/swe_any/generator/pipeline/post_feature.py \
  -d "${POST_DB}" \
  --max-concurrent 1 \
  --worker-processes 1 \
  --retry-failed 1 \
  --export-format jsonl \
  --export-path "${POST_RESULTS}"

echo "==================== swe_any feature pipeline: done ===================="
echo "Artifacts: ${SWE_ANY_OUTPUT_DIR}"
echo "Pre results: ${SWE_ANY_PRE_FEATURE_RESULTS}"
echo "Feature results: ${SWE_ANY_FEATURE_RESULTS}"
echo "Post results: ${POST_RESULTS}"
