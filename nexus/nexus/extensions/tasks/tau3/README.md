# TAU3 客服模拟评测

## 概述

TAU3（基于 [tau2-bench](https://github.com/sierra-research/tau2-bench)）评测模型在客服场景下的 Agent 能力：多轮对话、工具调用、知识检索、遵循策略等。

评测流程：Agent（被测模型）与 User Simulator（模拟用户）进行多轮对话，Agent 需要根据系统策略（system prompt）使用工具完成用户请求，最终由评估器检查数据库状态和对话质量来打分（0.0–1.0）。

### 4 个评测域

| 域 | base split | test split | 说明 |
|----|-----------|------------|------|
| airline | 50 | 20 | 航空公司客服：改签、退票、行李查询等 |
| retail | 114 | 40 | 零售客服：退货、换货、订单查询等 |
| telecom | 114 | 40 | 电信客服：话费、套餐、网络故障等 |
| banking_knowledge | 97 | 97 | 银行客服：信用卡申请、账户操作等（需知识库检索） |

## 前置条件

### 1. 安装依赖

```bash
uv venv
uv pip install -e ".[dev,extensions,tau3]"
```

### 2. 准备 tau2 数据

tau2 的 PyPI wheel **不包含评测数据文件**，需要手动 clone 并设置 `TAU2_DATA_DIR`：

```bash
git clone https://github.com/sierra-research/tau2-bench.git ./workspace/tau2
export TAU2_DATA_DIR=./workspace/tau2/data
```

> 如果不设置，运行时会报 `FileNotFoundError: .../data/tau2/domains/airline/tasks.json`。

### 3. 环境变量

批量评测通过环境变量传入配置。必填项：

```bash
# 被测模型（Agent）
export TAU3_AGENT_API_KEY="<your-api-key>"
export TAU3_AGENT_BASE_URL="http://litellm-sgp.woa.com"
export TAU3_AGENT_MODEL="gemini-3-flash-preview"

# 用户模拟器（建议用强模型以保证模拟质量）
export TAU3_USER_API_KEY="<your-api-key>"
export TAU3_USER_BASE_URL="http://litellm-sgp.woa.com"
export TAU3_USER_MODEL="gpt-5.2"

# 评测域和数据切分
export TAU3_DOMAIN="all"          # all | airline | retail | telecom | banking_knowledge
export TAU3_TASK_SPLIT="base"     # base（推荐） | test | train

# NL_ASSERTION 裁判模型（banking_knowledge 域需要，建议用强模型）
export TAU3_EVAL_CLIENT_NAME="openai"
export TAU3_EVAL_API_KEY="<your-api-key>"
export TAU3_EVAL_BASE_URL="http://litellm-sgp.woa.com"
export TAU3_EVAL_MODEL="gpt-5.2"
```

可选项：

```bash
export TAU3_RETRIEVAL_CONFIG="golden_retrieval"  # 知识检索策略（默认 golden_retrieval）
export TAU3_MAX_STEPS=200                        # 最大对话轮数
export TAU3_MAX_ERRORS=10                        # 最大工具调用错误数
export TAU3_SEED=42                              # 随机种子
# export TAU3_NUM_TASKS=10                         # 限制任务数（调试用）
export TAU3_SOLO_MODE=false                      # true 时不使用用户模拟器
```

### 使用 Taiji Client（Polaris 服务发现）

将 `TAU3_AGENT_CLIENT_NAME` 设为 `taiji`，并配置 Taiji 专属参数：

```bash
# Agent 使用 Taiji
export TAU3_AGENT_CLIENT_NAME=taiji
export TAU3_AGENT_MODEL="your-taiji-model-name"

# Taiji 公共参数（以下均有默认值，不设也可以）
# export TAU3_TAIJI_AUTH="Bearer 7auGXNATFSKl7dF"                          # 默认值
# export TAU3_TAIJI_WSID="11644"                                            # 默认值
# export TAU3_TAIJI_NS="Production"                                         # 默认值
# export TAU3_TAIJI_SERVICE="stream-server-online-sbs-11644"                # 默认值
# export TAU3_TAIJI_URL_PATTERN="openapi"                                  # openapi 或 v1
# export TAU3_TAIJI_TIMEOUT=120
# export TAU3_AGENT_ENABLE_THINKING=true                                   # 开启思考模式
# export TAU3_AGENT_TAIJI_SERVICE="custom-agent-service"                   # 覆盖 agent 的 service


# User simulator 仍用 OpenAI-compatible（推荐）
export TAU3_USER_CLIENT_NAME=openai
export TAU3_USER_API_KEY="<your-api-key>"
export TAU3_USER_BASE_URL="http://litellm-sgp.woa.com"
export TAU3_USER_MODEL="gpt-5.2"

# Eval 也可独立走 Taiji；不显式设置时建议默认走 openai
# export TAU3_EVAL_CLIENT_NAME=taiji
# export TAU3_EVAL_MODEL="your-eval-taiji-model"
# export TAU3_EVAL_ENABLE_THINKING=false
# export TAU3_EVAL_TAIJI_SERVICE="custom-eval-service"
```


## 运行方式

### 跑全量评测（推荐）

```bash
nexus run-batch \
  -g nexus/extensions/tasks/tau3/generator.py \
  -d ./outputs/tau3.db \
  --max-concurrent 32
```

### 跑单个域

```bash
TAU3_DOMAIN=airline TAU3_NUM_TASKS=5 \
nexus run-batch \
  -g nexus/extensions/tasks/tau3/generator.py \
  -d ./outputs/tau3_airline_debug.db \
  --max-concurrent 1
```

### 续跑（自动跳过已成功任务）

```bash
nexus run-batch \
  -d ./outputs/tau3.db \
  --max-concurrent 32
```

不需要再指定 `-g`，配置已锁定在数据库中。

## 查看结果

### 数据看板

```bash
nexus inspect-batch ./outputs/tau3.db
```

### 导出结果

```bash
# JSONL 格式
nexus export --format jsonl ./outputs/tau3.db

# 完整目录结构
nexus export --format full ./outputs/tau3.db
```

### SQL 查询

```bash
# 各域平均分
sqlite3 ./outputs/tau3.db "
  SELECT
    json_extract(metadata, '$.domain') as domain,
    ROUND(AVG(json_extract(task_output, '$.score')), 3) as avg_score,
    COUNT(*) as count
  FROM runs WHERE status='success'
  GROUP BY domain
"

# 总体分数
sqlite3 ./outputs/tau3.db "
  SELECT
    ROUND(AVG(json_extract(task_output, '$.score')), 3) as avg_score,
    SUM(CASE WHEN json_extract(task_output, '$.score') > 0 THEN 1 ELSE 0 END) as passed,
    COUNT(*) as total
  FROM runs WHERE status='success'
"

# 查看失败任务
sqlite3 ./outputs/tau3.db "
  SELECT run_id, json_extract(task_output, '$.score') as score,
         json_extract(task_output, '$.termination_reason') as reason,
         json_extract(task_output, '$.error') as error
  FROM runs WHERE status='success' AND json_extract(task_output, '$.score') = 0
  ORDER BY run_id
"
```

## 配置说明

### YAML 配置（config.example.yaml）

YAML 配置主要用于 `nexus run-single` 单任务调试，完整的字段参考见 `config.example.yaml`。

关键配置项：

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `task_config.domain` | 评测域 | `"all"` |
| `task_config.task_split` | 数据切分 | `"base"` |
| `task_config.retrieval_config` | 知识检索策略 | `"golden_retrieval"` |
| `task_config.eval_client_name` | NL_ASSERTION 裁判 client key | `"eval"` |
| `openai_clients.main` | 被测模型 | — |
| `openai_clients.user` | 用户模拟器模型 | — |
| `openai_clients.eval` | NL_ASSERTION 裁判模型 | — |

### retrieval_config 选项

banking_knowledge 域需要知识库检索。可用的检索策略：

| 策略 | 说明 | 推荐场景 |
|------|------|----------|
| `golden_retrieval` | 使用 ground truth 文档 | **推荐**，隔离检索影响，纯测 Agent 能力 |
| `bm25` | BM25 关键词检索 | 测试端到端效果（含检索） |
| `bm25_grep` | BM25 + grep 补充 | 改善 BM25 召回 |
| `openai_embeddings_grep` | OpenAI embedding + grep | 需要 OpenAI API |
| `qwen_embeddings_grep` | Qwen embedding + grep | 需要 Qwen API |

### 并发数建议

tau3 每个任务包含多轮 Agent-User 对话（每轮各调一次 LLM），并发数取决于 API RPM 限制：

| RPM 限制 | 建议并发数 |
|----------|-----------|
| 500 | 16–32 |
| 2000 | 64 |
| 5000+ | 128 |

## 评分说明

- **airline / retail / telecom**：评估器 replay Agent 的工具调用序列，检查数据库最终状态是否正确（DB score）
- **banking_knowledge**：DB score + 可选的 NL_ASSERTION score（LLM 裁判评判对话质量）
- 分数范围：0.0–1.0，其中 1.0 表示完全正确完成任务

## 注意事项

1. **用户模拟器建议用强模型**（如 gpt-5.2），弱模型可能生成不合理的用户行为导致 Agent 无法正常完成任务
2. **banking_knowledge 的 NL_ASSERTION 评估**通过 `openai_clients.eval` 走和 agent/user 相同的 client registry
3. **banking_knowledge 建议用 `golden_retrieval`** 排除检索质量的影响，聚焦测试 Agent 的工具调用和策略遵循能力
4. **高并发可能触发 RPM 限制**（429 错误），nexus 会自动重试，但建议控制并发数
