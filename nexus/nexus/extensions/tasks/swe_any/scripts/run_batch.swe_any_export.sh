#!/usr/bin/env bash
# Batch Export for SWE-Any (assemble → execution-grade → Harbor emit).
#
# Outer script sets DB / DB_PATH (and SWE_ANY_INSTANCES), then:
#   bash nexus/extensions/tasks/swe_any/scripts/run_batch.swe_any_export.sh
#
# Use an Export-only DB. Requires host issue.jsonl (from a prior Issue batch) under
# SWE_ANY_OUTPUT_DIR. DETERMINISTIC stage: no host LLM, so only GONGFENG_TOKEN is required.
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

export GONGFENG_TOKEN="${GONGFENG_TOKEN:?set GONGFENG_TOKEN (sandbox API token)}"
export SWE_ANY_INSTANCES="${SWE_ANY_INSTANCES:?set SWE_ANY_INSTANCES to a .jsonl file}"
export SWE_ANY_OUTPUT_DIR="${SWE_ANY_OUTPUT_DIR:-./outputs/swe_any}"

MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
DB="${DB:-${DB_PATH:-${SWE_ANY_OUTPUT_DIR%/}/export_batch.db}}"
EXPORT_PATH="${EXPORT_PATH:-${SWE_ANY_OUTPUT_DIR%/}/export_batch_results.jsonl}"
WORKER_PROCESSES="${WORKER_PROCESSES:-8}"
mkdir -p "$(dirname "${DB}")" "${SWE_ANY_OUTPUT_DIR}"

uv run nexus run-batch \
  -g nexus/extensions/tasks/swe_any/generator/stage/export.py \
  -d "${DB}" \
  --max-concurrent "${MAX_CONCURRENT}" \
  --worker-processes "${WORKER_PROCESSES}" \
  --retry-failed 1 \
  --export-format jsonl \
  --export-path "${EXPORT_PATH}"

echo "Batch DB:        ${DB}"
echo "Results (jsonl): ${EXPORT_PATH}"
echo "Harbor dataset:  ${SWE_ANY_OUTPUT_DIR%/}/harbor/<instance_id>/"
