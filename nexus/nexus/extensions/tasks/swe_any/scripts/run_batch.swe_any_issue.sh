#!/usr/bin/env bash
# Batch Issue for SWE-Any (Claude Code writes a non-leaking problem statement per feature).
#
# Outer script sets DB / DB_PATH (and SWE_ANY_INSTANCES), then:
#   bash nexus/extensions/tasks/swe_any/scripts/run_batch.swe_any_issue.sh
#
# Use an Issue-only DB. Requires host eval.jsonl (from a prior Eval batch) under SWE_ANY_OUTPUT_DIR.
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

export GONGFENG_TOKEN="${GONGFENG_TOKEN:?set GONGFENG_TOKEN (sandbox API token)}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY (host LLM for Claude Code)}"
export SWE_ANY_INSTANCES="${SWE_ANY_INSTANCES:?set SWE_ANY_INSTANCES to a .jsonl file}"
export SWE_ANY_OUTPUT_DIR="${SWE_ANY_OUTPUT_DIR:-./outputs/swe_any}"

MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
DB="${DB:-${DB_PATH:-${SWE_ANY_OUTPUT_DIR%/}/issue_batch.db}}"
EXPORT_PATH="${EXPORT_PATH:-${SWE_ANY_OUTPUT_DIR%/}/issue_batch_results.jsonl}"
WORKER_PROCESSES="${WORKER_PROCESSES:-8}"
mkdir -p "$(dirname "${DB}")" "${SWE_ANY_OUTPUT_DIR}"

uv run nexus run-batch \
  -g nexus/extensions/tasks/swe_any/generator/stage/issue.py \
  -d "${DB}" \
  --max-concurrent "${MAX_CONCURRENT}" \
  --worker-processes "${WORKER_PROCESSES}" \
  --retry-failed 1 \
  --export-format jsonl \
  --export-path "${EXPORT_PATH}"

echo "Batch DB:        ${DB}"
echo "Results (jsonl): ${EXPORT_PATH}"
echo "Archives under:  ${SWE_ANY_OUTPUT_DIR%/}/<repo_name>/issue.jsonl"
