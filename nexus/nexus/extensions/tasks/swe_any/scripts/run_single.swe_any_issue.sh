#!/usr/bin/env bash
# Run SWE-Any Issue on one instance (Claude Code writes a non-leaking problem statement).
#
#   SWE_ANY_INSTANCE_PATH=debug/swe_env/python/swe_any_smoke.jsonl \
#     bash nexus/extensions/tasks/swe_any/scripts/run_single.swe_any_issue.sh
#
# Requires Mask (mask.jsonl) and Eval (eval.jsonl) under SWE_ANY_OUTPUT_DIR.
#
# Only what config/stage/issue.yaml cannot supply itself lives here: secrets, the per-run
# instance, and the vertex model-name convention. Every other knob has an oc.env default there.
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

export GONGFENG_TOKEN="${GONGFENG_TOKEN:?set GONGFENG_TOKEN (sandbox API token)}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY (host LLM for Claude Code)}"
if [ "${OPENAI_CLIENT_NAME:-litellm}" = 'vertex_anthropic' ]; then
  export OPENAI_MODEL="${OPENAI_MODEL:-claude-opus-4-8}"
fi

export SWE_ANY_OUTPUT_DIR="${SWE_ANY_OUTPUT_DIR:-./outputs/swe_any}"
export SWE_ANY_INSTANCE_PATH="${SWE_ANY_INSTANCE_PATH:-}"
if [[ -z "${SWE_ANY_INSTANCE_PATH}" ]]; then
  export SWE_ANY_IMAGE="${SWE_ANY_IMAGE:?set SWE_ANY_IMAGE or SWE_ANY_INSTANCE_PATH}"
  export SWE_ANY_REPO_NAME="${SWE_ANY_REPO_NAME:-repo}"
  export SWE_ANY_REPO_DIR="${SWE_ANY_REPO_DIR:-/testbed}"
  export SWE_ANY_EVAL_PATH="${SWE_ANY_EVAL_PATH:-${SWE_ANY_OUTPUT_DIR%/}/${SWE_ANY_REPO_NAME}/eval.jsonl}"
  if [[ ! -f "${SWE_ANY_EVAL_PATH}" ]]; then
    echo "eval.jsonl not found at ${SWE_ANY_EVAL_PATH} — run scripts/run_single.swe_any_eval.sh first." >&2
    exit 1
  fi
fi

CONFIG="${CONFIG:-nexus/extensions/tasks/swe_any/config/stage/issue.yaml}"
OUTPUT="${OUTPUT:-${SWE_ANY_OUTPUT_DIR%/}/issue_result.json}"
mkdir -p "$(dirname "${OUTPUT}")"

uv run nexus run-single "${CONFIG}" -o "${OUTPUT}"

echo "Result:    ${OUTPUT}"
echo "Deliverable: ${SWE_ANY_OUTPUT_DIR%/}/<repo_name>/issue.jsonl (+ issue_agentic.jsonl, sgNN/issue_agentic.json)"
