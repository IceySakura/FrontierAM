#!/usr/bin/env bash
# Batch Select for SWE-Any (heuristic slices + Claude Code agentic pass).
#
# Outer script sets DB / DB_PATH (and SWE_ANY_INSTANCES), then:
#   bash nexus/extensions/tasks/swe_any/scripts/run_batch.swe_any_select.sh
#
# Use a Select-only DB (different from Parse) so run_ids do not collide.
# Requires host parse_graph.json under SWE_ANY_OUTPUT_DIR from a prior Parse batch.
#
# Only what must exist before Python starts lives here. The generator builds each
# item from config/stage/select.yaml (shared with run-single), merges
# SWE_ANY_CONFIG_PATCH over it when set, then merges config/client/<name>.yaml for
# the merged openai_clients.main.name — export any ${oc.env:...} the YAML
# references to override a default (OPENAI_CLIENT_NAME / SWE_ANY_MAX_SUBGRAPHS / …).
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

export GONGFENG_TOKEN="${GONGFENG_TOKEN:?set GONGFENG_TOKEN (sandbox API token)}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY (host LLM for Claude Code)}"
export SWE_ANY_INSTANCES="${SWE_ANY_INSTANCES:?set SWE_ANY_INSTANCES to a .jsonl file}"
# Optional subset / paging (empty = no filter) + a fixed seed draw for reproducibility.
# export SWE_ANY_INSTANCE_IDS="swe-env.v1.finos__symphony-bdk-python-284,/path/to/ids.json"
# export SWE_ANY_SKIP=0
# export SWE_ANY_LIMIT=0   # unset/0 = no limit
# export SWE_ANY_RNG_SEED=3407
# Optional task-knob patches, deep-merged over config/stage/select.yaml in order
# (comma-separated, later wins) so patches compose by permutation. Bare names
# resolve under config/custom/; a relative/absolute path also works:
# export SWE_ANY_CONFIG_PATCH=smoke.yaml
# export SWE_ANY_CONFIG_PATCH=smoke.yaml,vertex.yaml
# Mirrored in the config's oc.env default; the DB / EXPORT_PATH below are built from it,
# and Select reads each repo's parse_graph.json / parse_seed_policy.json from under it.
export SWE_ANY_OUTPUT_DIR="${SWE_ANY_OUTPUT_DIR:-./outputs/swe_any}"

# Peak concurrent sandboxes; each instance holds one for all of its agentic slices.
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
DB="${DB:-${DB_PATH:-${SWE_ANY_OUTPUT_DIR%/}/select_batch.db}}"
EXPORT_PATH="${EXPORT_PATH:-${SWE_ANY_OUTPUT_DIR%/}/select_batch_results.jsonl}"
# Shards the batch across processes and splits MAX_CONCURRENT between them. A sharded DB
# needs > 1 to resume, so keep this stable for a given DB.
WORKER_PROCESSES="${WORKER_PROCESSES:-8}"
mkdir -p "$(dirname "${DB}")" "${SWE_ANY_OUTPUT_DIR}"

uv run nexus run-batch \
  -g nexus/extensions/tasks/swe_any/generator/stage/select.py \
  -d "${DB}" \
  --max-concurrent "${MAX_CONCURRENT}" \
  --worker-processes "${WORKER_PROCESSES}" \
  --retry-failed 1 \
  --export-format jsonl \
  --export-path "${EXPORT_PATH}"

echo "Batch DB:        ${DB}"
echo "Results (jsonl): ${EXPORT_PATH}"
echo "Deliverable under:  ${SWE_ANY_OUTPUT_DIR%/}/<repo_name>/select.jsonl (+ {select_heuristic,select_agentic}.jsonl)"
