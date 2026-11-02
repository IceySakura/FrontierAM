# SWE Rubrics

Agent 自动生成评测 rubrics，LLM judge 对 patch 打分，输出归一化 [0,1] 分数。

## 三种运行模式

通过不同的 generator 文件选择模式，均使用 `nexus run-batch` 执行：

| 模式 | Generator | 用途 | 需要 Runtime |
|---|---|---|---|
| 只生成 | `agentic_patch/gen_generator.py` | 为每个 instance 生成 rubrics.yaml | 是 |
| 只评分 | `agentic_patch/eval_generator.py` | 用已有 rubrics 对 patch 打分 | 否 |
| 整合 | `agentic_patch/full_generator.py` | 生成 rubrics + 评分一条龙 | 是 |

## 运行方法

```bash
cd ${NEXUS_DIR}
source ${VENV_PATH}/bin/activate

# 设置环境变量（见下方完整列表）
export SWE_DATASET_PATH=...
export OPENAI_API_KEY=...
...

# 执行
uv run --active nexus run-batch \
    -g nexus/extensions/tasks/swe/rubrics/agentic_patch/<generator>.py \
    -d <output>.db \
    --max-concurrent <N> \
    --export-format jsonl \
    --export-path <output>.jsonl
```

### 只生成 rubrics

```bash
export SWE_DATASET_PATH=dataset.jsonl
export OPENAI_API_KEY=xxx
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=openai/gpt-4.1
export SWE_RUNTIME_PROVIDER_NAME=gongfeng
export SWE_RUNTIME_BASE_URL=http://hyrl-sandbox.prod.woa.com:8052
export SWE_RUNTIME_TOKEN=xxx
export SWE_RUNTIME_IMAGE=mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:latest
export SWE_LIMIT=10

uv run --active nexus run-batch \
    -g nexus/extensions/tasks/swe/rubrics/agentic_patch/gen_generator.py \
    -d rubric_gen.db --max-concurrent 5
```

### 只评分（三方 join）

需要三个数据源：SWE agent DB（提供 patch）+ rubric gen DB（提供 rubrics）+ dataset JSONL（提供 problem_statement）。按 instance_id 取交集。

```bash
export SWE_RESULTS_PATH=swe_agent.db
export RUBRIC_GEN_RESULTS_PATH=rubric_gen.db
export SWE_DATASET_PATH=dataset.jsonl
export OPENAI_API_KEY=xxx
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=openai/gpt-4.1

uv run --active nexus run-batch \
    -g nexus/extensions/tasks/swe/rubrics/agentic_patch/eval_generator.py \
    -d rubric_eval.db --max-concurrent 20
```

### 整合（生成 + 评分）

```bash
export SWE_RESULTS_PATH=swe_agent.db
export SWE_DATASET_PATH=dataset.jsonl
export OPENAI_API_KEY=xxx
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=openai/gpt-4.1
export SWE_RUNTIME_PROVIDER_NAME=gongfeng
export SWE_RUNTIME_TOKEN=xxx

uv run --active nexus run-batch \
    -g nexus/extensions/tasks/swe/rubrics/agentic_patch/full_generator.py \
    -d rubric_full.db --max-concurrent 5
```

## 环境变量完整列表

### 数据源

| 变量 | 说明 | 默认值 | 模式 |
|---|---|---|---|
| `SWE_DATASET_PATH` | 数据集 JSONL 路径（含 instance 信息、gold_patch、problem_statement） | 无，必须 | gen / eval / full |
| `SWE_RESULTS_PATH` | SWE agent batch 结果（`.db` 或 `.jsonl`，提供 patch_text）。DB 只读 status=success 的 run | 无，必须 | eval / full |
| `RUBRIC_GEN_RESULTS_PATH` | Rubric 生成 batch 结果（`.db` 或 `.jsonl`，提供 rubrics_yaml） | 无，必须 | eval |
| `SWE_INSTANCE_IDS` | 逗号分隔的 instance_id 过滤列表 | 全部 | gen / eval / full |
| `SWE_LIMIT` | 最大生成配置数 | 无限制 | gen / eval / full |

### LLM 配置

| 变量 | 说明 | 默认值 |
|---|---|---|
| `OPENAI_API_KEY` | LLM API key（生成和评分共用的默认值） | 无，必须 |
| `OPENAI_BASE_URL` | LLM API base URL | 无，必须 |
| `OPENAI_MODEL` | LLM 模型名（litellm 格式，如 `openai/gpt-4.1`） | 无，必须 |
| `RUBRIC_GEN_API_KEY` | 生成阶段 LLM API key（覆盖 OPENAI_API_KEY） | 同 OPENAI_API_KEY |
| `RUBRIC_GEN_BASE_URL` | 生成阶段 LLM base URL（覆盖 OPENAI_BASE_URL） | 同 OPENAI_BASE_URL |
| `RUBRIC_GEN_MODEL` | 生成阶段模型名（覆盖 OPENAI_MODEL） | 同 OPENAI_MODEL |
| `RUBRIC_EVAL_API_KEY` | 评分阶段 LLM API key（覆盖 OPENAI_API_KEY） | 同 OPENAI_API_KEY |
| `RUBRIC_EVAL_BASE_URL` | 评分阶段 LLM base URL（覆盖 OPENAI_BASE_URL） | 同 OPENAI_BASE_URL |
| `RUBRIC_EVAL_MODEL` | 评分阶段模型名（覆盖 OPENAI_MODEL） | 同 OPENAI_MODEL |

### Runtime 配置（gen / full 需要）

| 变量 | 说明 | 默认值 |
|---|---|---|
| `SWE_RUNTIME_PROVIDER_NAME` | Runtime provider 类型 | `gongfeng` |
| `SWE_RUNTIME_BASE_URL` | Runtime API 地址 | `http://hyrl-sandbox.prod.woa.com:8052` |
| `SWE_RUNTIME_TOKEN` | Runtime 认证 token | 空 |
| `SWE_RUNTIME_IMAGE` | Runtime 容器镜像 | `current` |
| `SWE_RUNTIME_TIMEOUT` | Runtime 请求超时（秒） | `7200` |
| `SWE_RUNTIME_CPU` | Runtime CPU 核数 | `2` |
| `SWE_RUNTIME_MEMORY` | Runtime 内存（GB） | `10` |
| `SWE_RPC_PROTOCOL` | RPC 协议 | `http` |

### Agent 配置

| 变量 | 说明 | 默认值 |
|---|---|---|
| `RUBRIC_AGENT_NAME` | Rubric 生成 agent 的 registry 名称 | `swe_agent_rubrics` |
| `SWE_MAX_ITERATIONS` | Agent 最大迭代数 | `50` |
| `RUBRIC_GEN_CLIENT_NAME` | 生成 agent 使用的 openai_client 名称 | `main` |
| `RUBRIC_EVAL_CLIENT_NAME` | 评分使用的 openai_client 名称 | `rubric_eval` |

## 输出字段

DB 中每条 run 的 `task_output` 包含：

| 字段 | 类型 | 说明 |
|---|---|---|
| `score` | float | 归一化分数 [0, 1]，= weighted_sum / max_possible_score |
| `item_scores` | dict | 逐条二值分数，如 `{"FC1": 1, "SA1": 0, ...}` |
| `weighted_sum` | float | 原始加权得分 = sum(score_i * weight_i) |
| `max_possible_score` | float | 满分加权得分 = sum(weight_i) |
| `rubrics_yaml` | str | 生成的 rubrics YAML 原文 |
| `rubrics` | dict | 解析后的结构化 rubrics（metadata + axes） |
| `patch_text` | str | 被评测的 patch |
| `raw_response` | str | LLM judge 原始返回（debug 用） |
| `gen_completed` | bool | rubric 生成是否成功 |
| `error` | str/null | 错误信息，null 表示成功 |

## 目录结构

```
rubrics/
├── README.md            ← 本文件
├── shared/              # 跨评测方法的公共工具
│   ├── schema.py        #   RubricSchema, RubricScores, JSON 解析
│   └── generator_utils.py #  DB 读取, env 解析, LLM/runtime 配置构建
│
└── agentic_patch/       # 评测方法：agent 生成 rubrics + LLM 评分 patch
    ├── prompts.py       #   prompt 模板（生成 + 评分）
    ├── gen_task.py      #   swe_rubric_gen_task（只生成）
    ├── eval_task.py     #   swe_rubric_eval_task（只评分）+ score_patch_against_rubrics()
    ├── task.py          #   swe_rubric_task（整合）
    ├── gen_generator.py #   batch generator: 只生成
    ├── eval_generator.py #  batch generator: 只评分（三方 join）
    └── full_generator.py #  batch generator: 整合
```

## 注意事项

- **DB vs JSONL**：`SWE_RESULTS_PATH` 和 `RUBRIC_GEN_RESULTS_PATH` 支持 `.db`（直读 batch DB，只取 success）和 `.jsonl`（导出文件），按扩展名自动识别
- **评分 retry**：LLM 返回的 JSON 解析失败时自动 retry 3 次
- **Runtime 镜像**：容器镜像需要包含最新的 nexus 代码（含 rubrics 模块），否则 agent tool 执行时会报 registry 错误
- **校验策略**：环境变量/路径缺失在 generator 阶段 fail fast；单条数据问题在 task 层 graceful fail（返回 error，不影响其他 run）
