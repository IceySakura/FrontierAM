#!/usr/bin/env bash
# Run SWE-Any Parse on one instance (SCIP index → parse_graph.json → seed policy).
#
#   SWE_ANY_INSTANCE_PATH=debug/swe_env/python/swe_any_smoke.jsonl \
#     bash nexus/extensions/tasks/swe_any/scripts/run_single.swe_any_parse.sh
#
# Or set SWE_ANY_IMAGE / SWE_ANY_REPO_NAME / SWE_ANY_REPO_DIR explicitly.
#
# Only what config/stage/parse.yaml cannot supply itself lives here: secrets, the
# per-run instance, and the vertex model-name convention. Every other knob already
# has an oc.env default in that file — export it in your shell to override.
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

export GONGFENG_TOKEN="${GONGFENG_TOKEN:?set GONGFENG_TOKEN (sandbox API token)}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY (host LLM for the seed-policy pass)}"
# The config's OPENAI_MODEL default is the litellm/bedrock name; vertex_anthropic wants
# the bare id, which one YAML default cannot express. (The batch path handles this
# via the generator's backend patch instead.)
if [ "${OPENAI_CLIENT_NAME:-litellm}" = 'vertex_anthropic' ]; then
  export OPENAI_MODEL="${OPENAI_MODEL:-claude-opus-4-8}"
fi

# Mirrors the config default; the OUTPUT path below is built from it.
export SWE_ANY_OUTPUT_DIR="${SWE_ANY_OUTPUT_DIR:-./outputs/swe_any}"
export SWE_ANY_INSTANCE_PATH="${SWE_ANY_INSTANCE_PATH:-}"
if [[ -z "${SWE_ANY_INSTANCE_PATH}" ]]; then
  # Without an instance row, the workload image has to be named explicitly.
  export SWE_ANY_IMAGE="${SWE_ANY_IMAGE:?set SWE_ANY_IMAGE or SWE_ANY_INSTANCE_PATH}"
  export SWE_ANY_REPO_NAME="${SWE_ANY_REPO_NAME:-repo}"
  export SWE_ANY_REPO_DIR="${SWE_ANY_REPO_DIR:-/testbed}"
fi

CONFIG="${CONFIG:-nexus/extensions/tasks/swe_any/config/stage/parse.yaml}"
OUTPUT="${OUTPUT:-${SWE_ANY_OUTPUT_DIR%/}/parse_result.json}"
mkdir -p "$(dirname "${OUTPUT}")"

uv run nexus run-single "${CONFIG}" -o "${OUTPUT}"

echo "Result:        ${OUTPUT}"
echo "Graph:         ${SWE_ANY_OUTPUT_DIR%/}/<repo_name>/parse_graph.json"
echo "Seed policy:   ${SWE_ANY_OUTPUT_DIR%/}/<repo_name>/parse_seed_policy.json"
