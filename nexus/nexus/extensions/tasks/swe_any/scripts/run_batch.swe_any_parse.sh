#!/usr/bin/env bash
# Batch Parse for SWE-Any (SCIP index → parse_graph.json → seed policy).
#
# Outer script sets DB / DB_PATH (and SWE_ANY_INSTANCES), then:
#   bash nexus/extensions/tasks/swe_any/scripts/run_batch.swe_any_parse.sh
#
# Prefer a Parse-only DB so Select run_ids do not collide.
#
# Only what must exist before Python starts lives here. The generator builds each
# item from config/stage/parse.yaml (shared with run-single), merges
# SWE_ANY_CONFIG_PATCH over it when set, then merges config/client/<name>.yaml for
# the merged openai_clients.main.name — export any ${oc.env:...} the YAML
# references to override a default (OPENAI_CLIENT_NAME / OPENAI_MODEL / …).
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

export GONGFENG_TOKEN="${GONGFENG_TOKEN:?set GONGFENG_TOKEN (sandbox API token)}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY (host LLM for the seed-policy pass)}"
export SWE_ANY_INSTANCES="${SWE_ANY_INSTANCES:?set SWE_ANY_INSTANCES to a .jsonl file}"
# Optional subset / paging (empty = no filter). SWE_ANY_INSTANCE_IDS takes literal
# instance_ids and/or a path to a JSON list of them.
# export SWE_ANY_INSTANCE_IDS="swe-env.v1.finos__symphony-bdk-python-284,/path/to/ids.json"
# export SWE_ANY_SKIP=0
# export SWE_ANY_LIMIT=0   # unset/0 = no limit
# Optional task-knob patches, deep-merged over config/stage/parse.yaml in order
# (comma-separated, later wins) so patches compose by permutation. Bare names
# resolve under config/custom/; a relative/absolute path also works:
# export SWE_ANY_CONFIG_PATCH=smoke.yaml
# export SWE_ANY_CONFIG_PATCH=smoke.yaml,vertex.yaml
# Mirrored in the config's oc.env default; the DB / EXPORT_PATH below are built from it.
export SWE_ANY_OUTPUT_DIR="${SWE_ANY_OUTPUT_DIR:-./outputs/swe_any}"

# Peak concurrent sandboxes; one sandbox per instance (index + seed-policy agent).
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
DB="${DB:-${DB_PATH:-${SWE_ANY_OUTPUT_DIR%/}/parse_batch.db}}"
EXPORT_PATH="${EXPORT_PATH:-${SWE_ANY_OUTPUT_DIR%/}/parse_batch_results.jsonl}"
# Shards the batch across processes and splits MAX_CONCURRENT between them. A sharded DB
# needs > 1 to resume, so keep this stable for a given DB.
WORKER_PROCESSES="${WORKER_PROCESSES:-8}"
mkdir -p "$(dirname "${DB}")" "${SWE_ANY_OUTPUT_DIR}"

uv run nexus run-batch \
  -g nexus/extensions/tasks/swe_any/generator/stage/parse.py \
  -d "${DB}" \
  --max-concurrent "${MAX_CONCURRENT}" \
  --worker-processes "${WORKER_PROCESSES}" \
  --retry-failed 1 \
  --export-format jsonl \
  --export-path "${EXPORT_PATH}"

echo "Batch DB:        ${DB}"
echo "Results (jsonl): ${EXPORT_PATH}"
echo "Graphs under:    ${SWE_ANY_OUTPUT_DIR%/}/<repo_name>/parse_graph.json"
echo "Seed policies:   ${SWE_ANY_OUTPUT_DIR%/}/<repo_name>/parse_seed_policy.json"
