#!/usr/bin/env bash
# Run SWE-Any Export on one instance (assemble → execution-grade → Harbor emit).
#
#   SWE_ANY_INSTANCE_PATH=debug/swe_env/python/swe_any_smoke.jsonl \
#     bash nexus/extensions/tasks/swe_any/scripts/run_single.swe_any_export.sh
#
# Requires Mask (mask.jsonl), Eval (eval.jsonl) and Issue (issue.jsonl) under SWE_ANY_OUTPUT_DIR.
# DETERMINISTIC stage: no host LLM, so only GONGFENG_TOKEN is required.
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

export GONGFENG_TOKEN="${GONGFENG_TOKEN:?set GONGFENG_TOKEN (sandbox API token)}"

export SWE_ANY_OUTPUT_DIR="${SWE_ANY_OUTPUT_DIR:-./outputs/swe_any}"
export SWE_ANY_INSTANCE_PATH="${SWE_ANY_INSTANCE_PATH:-}"
if [[ -z "${SWE_ANY_INSTANCE_PATH}" ]]; then
  export SWE_ANY_IMAGE="${SWE_ANY_IMAGE:?set SWE_ANY_IMAGE or SWE_ANY_INSTANCE_PATH}"
  export SWE_ANY_REPO_NAME="${SWE_ANY_REPO_NAME:-repo}"
  export SWE_ANY_REPO_DIR="${SWE_ANY_REPO_DIR:-/testbed}"
  export SWE_ANY_ISSUE_PATH="${SWE_ANY_ISSUE_PATH:-${SWE_ANY_OUTPUT_DIR%/}/${SWE_ANY_REPO_NAME}/issue.jsonl}"
  if [[ ! -f "${SWE_ANY_ISSUE_PATH}" ]]; then
    echo "issue.jsonl not found at ${SWE_ANY_ISSUE_PATH} — run scripts/run_single.swe_any_issue.sh first." >&2
    exit 1
  fi
fi

CONFIG="${CONFIG:-nexus/extensions/tasks/swe_any/config/stage/export.yaml}"
OUTPUT="${OUTPUT:-${SWE_ANY_OUTPUT_DIR%/}/export_result.json}"
mkdir -p "$(dirname "${OUTPUT}")"

uv run nexus run-single "${CONFIG}" -o "${OUTPUT}"

echo "Result:      ${OUTPUT}"
echo "Deliverable: ${SWE_ANY_OUTPUT_DIR%/}/<repo_name>/export.jsonl + ${SWE_ANY_OUTPUT_DIR%/}/harbor/<instance_id>/"
