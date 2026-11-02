#!/bin/bash

env_name=""
total_count_per_round=5000
batch_id=""

origin_options=("$@")

# 检查 --env xxx 是否传入, 更新ENV_NAME
while [[ $# -gt 0 ]]; do
  case $1 in
    --env)
      env_name="$2"
      shift 2
      ;;
    --total-count-per-round)
      total_count_per_round="$2"
      shift 2
      ;;
    --batch-id)
      batch_id="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

if [[ -z "$env_name" ]]; then
  echo "Usage: $0 --env <environment_name> [other options]"
  exit 1
fi

# Set batch_id if provided
if [[ -n "$batch_id" ]]; then
  export SWE_BATCH_ID="$batch_id"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WORK_DIR="./workspace/benchmark_${env_name}_${TIMESTAMP}"
mkdir -p "$WORK_DIR"

# Log file in the same directory as db files
LOG_FILE="${WORK_DIR}/benchmark_${TIMESTAMP}.log"
echo "LOG_FILE: ${LOG_FILE}"

# Activate virtual environment
source .venv/bin/activate

# SWE_RUNTIME_TAG=gerrylonwan_benchmark \
python -u tests/stress/swe_sandbox/run_benchmark.py \
  --instance-ids "sympy__sympy-13031,sphinx-doc__sphinx-8265,django__django-12125,django__django-11265,django__django-11490" \
  --rounds 1 --total-count-per-round ${total_count_per_round} --yes \
  --multiprocess --tasks-per-process 500 \
  --db-path "$WORK_DIR" \
  "${origin_options[@]}" >"$LOG_FILE" 2>&1