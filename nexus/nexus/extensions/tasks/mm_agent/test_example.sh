# Search Task Batch Runner Configuration Template
# Copy this file to .env.search and fill in your credentials
# Usage 1: source .env.search && uv run nexus run-batch ...
# Usage 2: python run_benchmarks.py -s -m 200 
# ============================================================================
# Required Configuration
# ============================================================================


export JUDGE_OPENAI_CLIENTS_NAME="litellm"
export JUDGE_OPENAI_BASE_URL=http://trpc-gpt-eval.production.polaris:8080/api/v3
export JUDGE_OPENAI_API_KEY="user:key?provider=doubao&timeout=600&model=doubao-seed-1-6-251015"
export JUDGE_OPENAI_MODEL=volcengine/doubao-seed-1-6-251015


# Reward Service
export SEARCH_EVAL_MODE=llm_judge

# Infer Porxy
export SEARCH_OPENAI_CLIENTS_NAME="openai" #infer_proxy or openai
# # export SEARCH_OPENAI_BASE_URL="http://infer-proxy-yb-test.production.polaris:8000/openapi"
# # export SEARCH_OPENAI_MODEL="A3B_innodata_step440_GRM" 
export SEARCH_OPENAI_BASE_URL="http://llm-api.model-eval.woa.com/v1"
export SEARCH_OPENAI_MODEL="api_naci_default_gemini-3-flash-preview"
# export SEARCH_OPENAI_MODEL="api_openai_gpt-5.2"
# export SEARCH_OPENAI_MODEL="api_naci_default_gemini-3-pro-preview"
export SEARCH_OPENAI_API_KEY="user:key"
# export SEARCH_OPENAI_MODEL="api_openai_gpt-5.2"


export SEARCH_TEMPERATURE=1
export SEARCH_THINKING_TYPE=enabled
export SEARCH_REASONING_EFFORT=high
export SEARCH_INTERLEAVED_THINKING=true
export SEARCH_MAX_TOKENS=32768
# export SEARCH_MAX_TOKENS=131072
export SEARCH_ENABLE_SUMMARY=true

# Tool Configuration - Google Search (REQUIRED)
export SEARCH_SERPER_URL='http://trpc-gpt-eval.production.polaris:8080/search'
export T2I_SEARCH_SERPER_URL='http://trpc-gpt-eval.production.polaris:8080/images'
export I2I_SEARCH_SERPER_URL='http://trpc-gpt-eval.production.polaris:8080/lens'
export SEARCH_SERPER_APP_ID="user"
export SEARCH_SERPER_APP_KEY="key"

export IMGBB_API_KEY="xxx"

# Tool Configuration - Visit Tool (Jina)
export DEFAULT_JINA_URL="http://trpc-gpt-eval.production.polaris:8080/api/jina_ai/search"
export DEFAULT_JINA_APP_ID="user"
export DEFAULT_JINA_APP_KEY="key"
export DEFAULT_JINA_PROVIDER="jina_ai"
export DEFAULT_JINA_MODEL="api_doubao_deepseek-v3-2-251201"
export DEFAULT_JINA_MODEL_APP_ID="user"
export DEFAULT_JINA_MODEL_APP_KEY="key"
export TRAINING_LENGTH_EXCEEDED="True"

# Tool Configuration - Code Interpreter
export PYTHON_PATH="/opt/conda/envs/torch-base/bin/python3"
export CODE_INTERPRETER_TIMEOUT=30
export CODE_INTERPRETER_WORK_DIR="/tmp/code_interpreter"

export SEARCH_AGENT_NAME=mm_search_agent
export SEARCH_MAX_ITERATIONS=30
export SEARCH_AGENT_MAX_LLM_RETRIES=15
export SEARCH_SKIP=0
export SEARCH_LIMIT=100000000
export SEARCH_TIMEOUT_PER_RUN=3600
export SEARCH_RUN_ORACLE=false
export IS_TRAINING=false
export JUDGE_USE_TOOL_CALLING=true


#babyvision
export SEARCH_DATASET_PATH=/apdcephfs_cq8/share_1611098/shitaoxiao/agent/evaluation/local_test/datasets/babyvision/transformed.parquet

# export NOT_USE_TOOL=true
# uv run nexus run-batch -g /apdcephfs_cq8/share_1611098/shitaoxiao/agent/nexus/nexus/extensions/tasks/mm_agent/generator.py -d /test/gemini3flash/babyvision_notool.db --max-concurrent 64

export NOT_USE_TOOL=false
uv run nexus run-batch -g /apdcephfs_cq8/share_1611098/shitaoxiao/agent/nexus/nexus/extensions/tasks/mm_agent/generator.py -d /test/gemini3flash/babyvision.db --max-concurrent 64



python nexus/extensions/tasks/search/compute_metrics.py --db /test/gemini3flash/babyvision.db
