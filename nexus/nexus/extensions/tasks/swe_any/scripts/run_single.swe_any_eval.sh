#!/usr/bin/env bash
# Run SWE-Any Eval on one instance (draft test-deletion + Claude Code test-oracle).
#
#   SWE_ANY_INSTANCE_PATH=debug/swe_env/python/swe_any_smoke.jsonl \
#     bash nexus/extensions/tasks/swe_any/scripts/run_single.swe_any_eval.sh
#
# Requires Mask to have written mask.jsonl (and Parse parse_graph.json) under
# SWE_ANY_OUTPUT_DIR.
#
# Only what config/stage/eval.yaml cannot supply itself lives here: secrets, the
# per-run instance, and the vertex model-name convention. Every other knob already
# has an oc.env default in that file — export it in your shell to override.
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

export GONGFENG_TOKEN="${GONGFENG_TOKEN:?set GONGFENG_TOKEN (sandbox API token)}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY (host LLM for Claude Code)}"
# The config's OPENAI_MODEL default is the litellm/bedrock name; vertex_anthropic wants
# the bare id, which one YAML default cannot express. (The batch path handles this
# via the generator's backend patch instead.)
if [ "${OPENAI_CLIENT_NAME:-litellm}" = 'vertex_anthropic' ]; then
  export OPENAI_MODEL="${OPENAI_MODEL:-claude-opus-4-8}"
fi

# Mirrors the config default; the OUTPUT and input paths below are built from it.
export SWE_ANY_OUTPUT_DIR="${SWE_ANY_OUTPUT_DIR:-./outputs/swe_any}"
export SWE_ANY_INSTANCE_PATH="${SWE_ANY_INSTANCE_PATH:-}"
if [[ -z "${SWE_ANY_INSTANCE_PATH}" ]]; then
  # Without an instance row, the workload image and the inputs have to be named explicitly.
  export SWE_ANY_IMAGE="${SWE_ANY_IMAGE:?set SWE_ANY_IMAGE or SWE_ANY_INSTANCE_PATH}"
  export SWE_ANY_REPO_NAME="${SWE_ANY_REPO_NAME:-repo}"
  export SWE_ANY_REPO_DIR="${SWE_ANY_REPO_DIR:-/testbed}"
  export SWE_ANY_GRAPH_PATH="${SWE_ANY_GRAPH_PATH:-${SWE_ANY_OUTPUT_DIR%/}/${SWE_ANY_REPO_NAME}/parse_graph.json}"
  export SWE_ANY_MASK_PATH="${SWE_ANY_MASK_PATH:-${SWE_ANY_OUTPUT_DIR%/}/${SWE_ANY_REPO_NAME}/mask.jsonl}"
  if [[ ! -f "${SWE_ANY_MASK_PATH}" ]]; then
    echo "mask.jsonl not found at ${SWE_ANY_MASK_PATH} — run scripts/run_single.swe_any_mask.sh first." >&2
    exit 1
  fi
fi

CONFIG="${CONFIG:-nexus/extensions/tasks/swe_any/config/stage/eval.yaml}"
OUTPUT="${OUTPUT:-${SWE_ANY_OUTPUT_DIR%/}/eval_result.json}"
mkdir -p "$(dirname "${OUTPUT}")"

uv run nexus run-single "${CONFIG}" -o "${OUTPUT}"

echo "Result:    ${OUTPUT}"
echo "Deliverable: ${SWE_ANY_OUTPUT_DIR%/}/<repo_name>/eval.jsonl (+ {eval_heuristic,eval_agentic}.jsonl, sgNN/{eval_heuristic,eval_agentic}.json)"
