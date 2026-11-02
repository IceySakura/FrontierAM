#!/usr/bin/env bash
# Run the whole SWE-Any pipeline over one instances JSONL, one batch stage after another:
#
#   Parse → Select → Mask → Eval → Issue → Export
#
# Each stage is just its own run_batch.swe_any_<stage>.sh (own per-stage DB under
# SWE_ANY_OUTPUT_DIR), invoked in order; a stage that fails aborts the chain (set -e). Stages read
# the prior stage's host artifacts, so ordering is the whole point.
#
#   SWE_ANY_INSTANCES=./debug/swe_env.jsonl \
#   GONGFENG_TOKEN=... OPENAI_API_KEY=... \
#     bash nexus/extensions/tasks/swe_any/scripts/run_swe_any.sh
#   # or pass the instances file as $1:
#     bash nexus/extensions/tasks/swe_any/scripts/run_swe_any.sh ./debug/swe_env.jsonl
#
# Run a subset (later stages need the earlier artifacts to already be on disk):
#   STAGES="issue export" bash .../run_swe_any.sh
#
# Knobs are the same env every stage reads (all optional): SWE_ANY_OUTPUT_DIR, MAX_CONCURRENT,
# WORKER_PROCESSES, SWE_ANY_CONFIG_PATCH, SWE_ANY_INSTANCE_IDS / _SKIP / _LIMIT / _LANGUAGE, and the
# per-stage SWE_ANY_*_USE_AGENT / timeouts — export them once here and they apply to every stage.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Instances: $1 wins, else SWE_ANY_INSTANCES.
if [[ -n "${1:-}" ]]; then
  export SWE_ANY_INSTANCES="$1"
fi
export SWE_ANY_INSTANCES="${SWE_ANY_INSTANCES:?set SWE_ANY_INSTANCES (or pass the .jsonl as \$1)}"
export SWE_ANY_OUTPUT_DIR="${SWE_ANY_OUTPUT_DIR:-./outputs/swe_any}"

# Which stages to run, in order (space-separated). Default = the full pipeline.
STAGES="${STAGES:-parse select mask eval issue export}"

# Secrets: GONGFENG_TOKEN always (sandbox); OPENAI_API_KEY whenever an LLM stage runs
# (everything but Export). Fail fast here rather than mid-chain.
export GONGFENG_TOKEN="${GONGFENG_TOKEN:?set GONGFENG_TOKEN (sandbox API token)}"
for stage in ${STAGES}; do
  if [[ "${stage}" != 'export' ]]; then
    export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY (host LLM for the ${stage} agent)}"
    break
  fi
done

for stage in ${STAGES}; do
  script="${SCRIPT_DIR}/run_batch.swe_any_${stage}.sh"
  if [[ ! -f "${script}" ]]; then
    echo "unknown stage '${stage}' (no ${script})" >&2
    exit 2
  fi
  echo "==================== swe_any: ${stage} ===================="
  # nexus run-batch returns 1 if ANY instance failed. At scale, per-instance HARD failures
  # (scip-python can't index a repo, an agent hard-errors, a sandbox drops) are EXPECTED and must
  # NOT abort the pipeline: downstream stages only consume the instances that produced artifacts.
  # So tolerate a stage's non-zero exit (log it) instead of letting `set -e` kill the chain. A true
  # soft fail (agent give-up -> warning, completed/success stay true) never counts as a failure here.
  if ! bash "${script}"; then
    echo "WARN: stage '${stage}' exited non-zero (partial per-instance failures); continuing" >&2
  fi
done

echo "==================== swe_any: done (${STAGES}) ===================="
echo "Artifacts under: ${SWE_ANY_OUTPUT_DIR%/}/<repo_name>/  (+ harbor/<instance_id>/ from export)"
