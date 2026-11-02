#!/bin/bash

# 批量执行 test.jsonl 的脚本

# ========== 路径配置 ==========
# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 获取项目根目录 (脚本位于 nexus/extensions/tasks/swe/tests/ 下)
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

# 切换到项目根目录执行
cd "${PROJECT_ROOT}"

# ========== 基础配置 ==========
# 数据集路径 (使用脚本所在目录)
export SWE_DATASET_PATH="${SCRIPT_DIR}/test.jsonl"

# 使用 golden patch (Oracle 模式)
export SWE_RUN_ORACLE=1

# ========== Runtime 配置 ==========
# Runtime provider 类型
export SWE_RUNTIME_PROVIDER_NAME="gongfeng"

# Gongfeng Runtime API 地址
export SWE_RUNTIME_BASE_URL="http://hyrl-sandbox.test.woa.com:8052"

# Runtime 镜像 tag (与你的 swe_task.yaml 保持一致)
# export SWE_RUNTIME_TAG="feature-fix_swe_turing"
# check whether the runtime tag is set
if [ -z "${SWE_RUNTIME_TAG}" ]; then
  # get current git branch name and set to SWE_RUNTIME_TAG
  # 将分支名中的 / 替换为 - (对齐流水线)
  SWE_RUNTIME_TAG=$(git branch --show-current | tr '/' '-')
  export SWE_RUNTIME_TAG
fi

# Runtime 超时时间 (秒)
export SWE_RUNTIME_TIMEOUT=1800

# ========== 可选：分批运行配置 ==========
# 跳过前 N 条
# export SWE_SKIP=0

# 最多运行 N 条
# export SWE_LIMIT=10

# 指定运行特定 instance_ids (逗号分隔)
# export SWE_INSTANCE_IDS="sympy__sympy-27555,pylint-dev__pylint-10344"

# ========== Agent 配置 (非 Oracle 模式时使用) ==========
# export SWE_AGENT_NAME="swe_agent"
# export SWE_MAX_ITERATIONS=5
# export OPENAI_API_KEY="your-api-key"
# export OPENAI_BASE_URL="https://api.openai.com/v1"
# export OPENAI_MODEL="gpt-4.1"

# ========== 执行批量任务 ==========
uv run nexus run-batch \
  -g "${PROJECT_ROOT}/nexus/extensions/tasks/swe/generator.py" \
  -d "${PROJECT_ROOT}/workspace/batch_test.db" \
  --max-concurrent 10