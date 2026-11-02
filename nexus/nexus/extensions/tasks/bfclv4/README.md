# BFCL v4 函数调用评测

## 概述

BFCL v4（[Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)）评测模型的函数调用（function calling）能力：工具选择、参数提取、多轮工具调用、不相关查询拒绝等。

评测流程：直接向被测模型发送带有工具定义的请求，收集模型返回的 tool_calls，由 BFCL 的 checker 对比 ground truth 打分（0 或 1）。 **不需要 Agent**——测的是模型原生的 function calling 能力。

### 评测类别（20+ categories，~5800 条）

| 分组 | 类别 | 数量 | 说明 |
|------|------|------|------|
| **Single-Turn** | simple_python, simple_java, simple_javascript | 550 | 单函数调用，不同语言 |
| | multiple | 200 | 选择正确函数（多个候选） |
| | parallel | 200 | 并行调用多个函数 |
| | parallel_multiple | 200 | 并行 + 多候选 |
| | irrelevance | 240 | 模型应拒绝调用（无相关函数） |
| **Live** | live_simple, live_multiple, live_parallel, live_parallel_multiple | 1351 | 真实 API 的函数调用 |
| | live_irrelevance, live_relevance | 900 | 相关性/不相关性检测 |
| **Multi-Turn** | multi_turn_base | 200 | 多轮对话中的工具调用 |
| | multi_turn_miss_func, multi_turn_miss_param | 400 | 缺失函数/参数时的鲁棒性 |
| | multi_turn_long_context | 200 | 长上下文多轮 |
| **Agentic** | web_search_base, web_search_no_snippet | 200 | 搜索 Agent |
| | memory_kv, memory_vector, memory_rec_sum | 576 | 记忆 Agent（按 scenario 并行，scenario 内串行） |

## 前置条件

### 1. 安装依赖

```bash
uv venv
uv pip install -e ".[dev,extensions,bfclv4]"
```

### 2. 环境变量

批量评测通过环境变量传入配置。 必填项：

```bash
# 被测模型
export BFCL_OPENAI_BASE_URL="http://litellm-cn.woa.com"
export BFCL_OPENAI_API_KEY="<your-api-key>"
export BFCL_OPENAI_MODEL="glm-5"

# 评测范围
export BFCL_CATEGORIES="all_scoring"   # 见下方「评测范围枚举」
```

可选项：

```bash
export BFCL_IS_FC_MODEL="true"          # 是否使用 FC 模式（默认 true）
export BFCL_TEMPERATURE="0"             # 采样温度（默认 0）
export BFCL_SKIP="0"                    # 每个类别跳过的条目数
# export BFCL_LIMIT="10"                  # 每个类别最多跑的条目数（调试用）
export BFCL_OPENAI_CLIENTS_NAME="openai"  # OpenAI client 注册名
export BFCL_MODEL_RESULT_DIR="./workspace/bfcl_memory"  # memory snapshot 存储目录
```

### 使用 Taiji Client（Polaris 服务发现）

将 `BFCL_OPENAI_CLIENTS_NAME` 设为 `taiji`，并配置 Taiji 专属参数：

```bash
# 切换到 Taiji client
export BFCL_OPENAI_CLIENTS_NAME="taiji"
export BFCL_OPENAI_MODEL="your-taiji-model-name"

# Taiji 公共参数（以下均有默认值，不设也可以）
# export BFCL_TAIJI_AUTH="Bearer 7auGXNATFSKl7dF"                          # 默认值
# export BFCL_TAIJI_WSID="11644"                                            # 默认值
# export BFCL_TAIJI_NS="Production"                                         # 默认值
# export BFCL_TAIJI_SERVICE="stream-server-online-sbs-11644"                # 默认值
# export BFCL_TAIJI_URL_PATTERN="openapi"                                  # openapi 或 v1
# export BFCL_TAIJI_TIMEOUT=120
# export BFCL_TAIJI_ENABLE_THINKING=true                                   # 开启思考模式
```

使用 Taiji 时不需要设 `BFCL_OPENAI_BASE_URL` 和 `BFCL_OPENAI_API_KEY`（由 Polaris 服务发现自动解析地址）。

> **注意**：Taiji client 依赖 `polaris-cpp-py` 包。 安装方式：`uv pip install /path/to/polaris-cpp-py-*.tar.gz`

### 评测范围枚举

| 枚举值 | 类别数 | 条目数 | 说明 |
|--------|--------|--------|------|
| `all_scoring` | 19 | ~4641 | **推荐**。 不含 memory，适合常规评测 |
| `all_scoring_with_memory` | 22 | ~5217 | 含 memory（576 条），需 `--timeout 1800` |
| `single_turn` | 7 | ~1390 | 单轮 FC 类别（含 irrelevance） |
| `multi_turn` | 4 | ~800 | 多轮对话 |
| `agentic` | 5 | ~776 | web_search + memory |
| `web_search` | 2 | ~200 | 仅搜索类别 |
| `memory` | 3 | ~576 | 仅记忆类别 |
| `live` | 6 | ~2251 | Live API 类别 |

> **为什么 `all_scoring` 不含 memory？** memory 类别的测评成本显著高于其他类别：每条需要多步推理（~1-2 分钟/条），scenario 内强制串行，memory_vector 还会加载 faiss 索引占用数 GB 内存。 建议日常评测用 `all_scoring`，需要完整报告时再用 `all_scoring_with_memory`。

## 运行方式

### 跑全量评测（推荐）

```bash
nexus run-batch \
  -g nexus/extensions/tasks/bfclv4/generator.py \
  -d ./outputs/bfcl.db \
  --max-concurrent 64 \
  --timeout 600 \
  2>&1 | tee ./outputs/bfcl.log
```

### 含 memory 的全量评测

```bash
BFCL_CATEGORIES=all_scoring_with_memory \
nexus run-batch \
  -g nexus/extensions/tasks/bfclv4/generator.py \
  -d ./outputs/bfcl_full.db \
  --max-concurrent 32 \
  --timeout 1800 \
  2>&1 | tee ./outputs/bfcl_full.log
```

### 跑部分类别（调试）

```bash
BFCL_CATEGORIES=simple_python BFCL_LIMIT=5 \
nexus run-batch \
  -g nexus/extensions/tasks/bfclv4/generator.py \
  -d ./outputs/bfcl_debug.db \
  --max-concurrent 5 \
  --timeout 120 \
  2>&1 | tee ./outputs/bfcl_debug.log
```

### 续跑（自动跳过已成功任务）

```bash
nexus run-batch \
  -d ./outputs/bfcl.db \
  --max-concurrent 32
```

不需要再指定 `-g`，配置已锁定在数据库中。

## 查看结果

### 数据看板

```bash
nexus inspect-batch ./outputs/bfcl.db
```

### 导出结果

```bash
# JSONL 格式
nexus export --format jsonl ./outputs/bfcl.db

# 完整目录结构
nexus export --format full ./outputs/bfcl.db
```

### SQL 查询

```bash
# 各类别准确率
sqlite3 ./outputs/bfcl.db "
  SELECT
    json_extract(metadata, '$.test_category') as category,
    ROUND(AVG(json_extract(task_output, '$.score')), 3) as accuracy,
    COUNT(*) as count
  FROM runs WHERE status='success'
  GROUP BY category
  ORDER BY accuracy DESC
"

# 总体分数
sqlite3 ./outputs/bfcl.db "
  SELECT
    ROUND(AVG(json_extract(task_output, '$.score')), 3) as avg_score,
    SUM(CASE WHEN json_extract(task_output, '$.score') = 1.0 THEN 1 ELSE 0 END) as passed,
    COUNT(*) as total
  FROM runs WHERE status='success'
"

# 查看失败任务详情
sqlite3 ./outputs/bfcl.db "
  SELECT run_id,
         json_extract(task_output, '$.score') as score,
         json_extract(task_output, '$.error') as error
  FROM runs WHERE status='success' AND json_extract(task_output, '$.score') = 0
  LIMIT 20
"

# 超时任务
sqlite3 ./outputs/bfcl.db "
  SELECT run_id, error FROM runs WHERE status='timeout'
"
```

## 架构说明

```
nexus/extensions/tasks/bfclv4/
├── generator.py      加载 BFCL 数据，populate 预处理，生成 BatchItem
│                     （web_search: show_snippet, memory: model_result_dir, multi_turn: function doc）
├── task.py           @task_registry.register('bfcl')
│   ├─ handler.py     NexusOpenAIHandler（BaseHandler 子类，用 sync OpenAI client）
│   │   └─ asyncio.to_thread(handler.inference(...))
│   └─ evaluator.py   路由到 BFCL checker
│       ├─ single_turn → ast_checker
│       ├─ relevance   → decode + 格式检查
│       ├─ multi_turn  → multi_turn_checker
│       └─ agentic     → agentic_checker
├── types.py          BFCLTaskInput / BFCLTaskOutput / 类别常量
└── README.md
```

### 关键设计

- **无 Agent**：BFCL 测原生 FC 能力，task 直接调 LLM，不走 Nexus Agent 流程
- **Sync→Async 桥接**：BFCL 的 BaseHandler 是同步的，通过 `asyncio.to_thread()` 包装实现批量并发
- **deepcopy(test_entry)**：BFCL 在 inference 过程中会修改 test_entry，必须深拷贝
- **Ground truth 随条目传递**：避免 task worker 依赖 BFCL 数据路径
- **输出净化（_sanitize_output）**：返回前用 `json.loads(json.dumps())` 将 model_responses / eval_result 转为纯 JSON 基本类型，防止框架 observe 的 `serialize_object()` 递归遍历 BFCL 内部对象导致内存爆炸
- **共享 OpenAI client**：模块级单例，避免 `SyncHttpxClientWrapper.__del__` RecursionError
- **动态模型名注入**：未知模型名自动注入 `MODEL_CONFIG_MAPPING`，根据 `is_fc_model` 决定 `underscore_to_dot`
- **BFCL Java 类型修复**：monkey-patch `JAVA_TYPE_CONVERSION` 和 `java_type_converter` 修复小写 `'string'` 缺失 bug

## 并发策略

| 类别 | 并发方式 | 说明 |
|------|---------|------|
| single_turn / live | 全速并发 | 单次 API 调用，推荐 32–64 并发 |
| multi_turn | 全速并发 | 多步推理但无状态依赖，推荐 16–32 并发 |
| agentic (web_search) | 全速并发 | 搜索 API 调用，推荐 10–32 并发 |
| agentic (memory) | **按 scenario 并行，scenario 内串行** | task 层 per-scenario `asyncio.Lock` 自动控制 |

### Memory 并发详解

Memory 类别（memory_kv / memory_vector / memory_rec_sum）每个有 5 个 scenario（customer / finance / healthcare / notetaker / student），共 15 个独立执行链。

**同一 scenario** 内的条目通过 snapshot 文件传递状态，必须串行执行：
```
prereq-0 → prereq-1 → ... → prereq-N → eval-0 → eval-1 → ...
```

**不同 scenario** 完全独立（读写不同 snapshot 文件），可以并行。 3 个类别 × 5 个 scenario = 最多 **15 路并行**。

Snapshot 文件存储在 `./workspace/bfcl_memory/`（可通过 `BFCL_MODEL_RESULT_DIR` 自定义）：
```
workspace/bfcl_memory/agentic/memory/kv/memory_snapshot/
├── customer_final.json         ← 每条执行完覆写
├── notetaker_final.json
├── ...
└── prereq_checkpoints/         ← prereq 单独快照
    ├── memory_kv_prereq_0-customer-0.json
    └── ...
```

> **注意**：`--timeout` 对 memory 条目需设为 1800 秒以上。 因为 batch 层 timeout 包含排队等锁的时间，而 memory 条目必须等同 scenario 的前序条目全部完成后才能开始执行。

## 已知限制

1. **format_sensitivity** 类别（5200 条）未包含在 `all_scoring` 中，需单独指定
2. **memory 类别**的 snapshot 目录在续跑时需要保留，否则非首条 prereq 会从空状态开始
