#!/usr/bin/env bash
# Batch Mask for SWE-Any (per-feature draft mask + Claude Code refine).
#
# Outer script sets DB / DB_PATH (and SWE_ANY_INSTANCES), then:
#   bash nexus/extensions/tasks/swe_any/scripts/run_batch.swe_any_mask.sh
#
# Use a Mask-only DB (different from Parse/Select) so run_ids do not collide.
# Requires host select.jsonl (from a prior Select batch) under SWE_ANY_OUTPUT_DIR.
#
# Only what must exist before Python starts lives here. The generator builds each
# item from config/stage/mask.yaml (shared with run-single), merges
# SWE_ANY_CONFIG_PATCH over it when set, then merges config/client/<name>.yaml for
# the merged openai_clients.main.name — export any ${oc.env:...} the YAML
# references to override a default (OPENAI_CLIENT_NAME / SWE_ANY_MAX_FEATURES / …).
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

export GONGFENG_TOKEN="${GONGFENG_TOKEN:?set GONGFENG_TOKEN (sandbox API token)}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY (host LLM for Claude Code)}"
export SWE_ANY_INSTANCES="${SWE_ANY_INSTANCES:?set SWE_ANY_INSTANCES to a .jsonl file}"
# Optional subset / paging (empty = no filter).
# export SWE_ANY_INSTANCE_IDS="swe-env.v1.finos__symphony-bdk-python-284,/path/to/ids.json"
# export SWE_ANY_SKIP=0
# export SWE_ANY_LIMIT=0   # unset/0 = no limit
# export SWE_ANY_MAX_FEATURES=0   # 0 = every resolved feature
# Optional task-knob patches, deep-merged over config/stage/mask.yaml in order
# (comma-separated, later wins). Bare names resolve under config/custom/:
# export SWE_ANY_CONFIG_PATCH=smoke.yaml
# Mirrored in the config's oc.env default; the DB / EXPORT_PATH below are built from it,
# and Mask reads each repo's parse_graph.json / select.jsonl from under it.
export SWE_ANY_OUTPUT_DIR="${SWE_ANY_OUTPUT_DIR:-./outputs/swe_any}"

# Peak concurrent sandboxes; each instance holds one for all of its feature masks.
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
DB="${DB:-${DB_PATH:-${SWE_ANY_OUTPUT_DIR%/}/mask_batch.db}}"
EXPORT_PATH="${EXPORT_PATH:-${SWE_ANY_OUTPUT_DIR%/}/mask_batch_results.jsonl}"
# Shards the batch across processes and splits MAX_CONCURRENT between them. A sharded DB
# needs > 1 to resume, so keep this stable for a given DB.
WORKER_PROCESSES="${WORKER_PROCESSES:-8}"
mkdir -p "$(dirname "${DB}")" "${SWE_ANY_OUTPUT_DIR}"

uv run nexus run-batch \
  -g nexus/extensions/tasks/swe_any/generator/stage/mask.py \
  -d "${DB}" \
  --max-concurrent "${MAX_CONCURRENT}" \
  --worker-processes "${WORKER_PROCESSES}" \
  --retry-failed 1 \
  --export-format jsonl \
  --export-path "${EXPORT_PATH}"

echo "Batch DB:        ${DB}"
echo "Results (jsonl): ${EXPORT_PATH}"
echo "Archives under:  ${SWE_ANY_OUTPUT_DIR%/}/<repo_name>/mask.jsonl"
