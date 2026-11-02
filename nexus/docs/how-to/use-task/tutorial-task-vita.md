# VitaBench 任务说明

本文说明 `vita_task` 的完整任务配置，并给出最小运行示例。 `openai_clients`、`processors` 已在 README 中说明，本文不再展开。

## 任务概览

- 任务名：`vita_task`，用于运行 VitaBench 生活服务场景的智能体模拟与评测。
- 关键组件：通过三方协作循环（Agent ↔ User Simulator ↔ Environment）完成模拟对话，随后使用滑动窗口 + Rubric 的方式对轨迹进行 LLM 评估并产出评分。
- Agent：`vita_task` 仅支持 `vita_agent` 作为推理代理，不可替换为其他 agent。
- 主要文件：`workspace/vita.yaml`（任务配置）、`nexus/extensions/tasks/vita/task.py`（任务逻辑）、`nexus/extensions/agents/vita_agent/agent.py`（代理逻辑）。

## 支持的领域（Domain）

VitaBench 涵盖三大生活服务领域，可单独运行或组合为跨领域任务：

| 领域 | 说明 | 工具数 |
|------|------|--------|
| `delivery` | 外卖配送 | 20 |
| `instore` | 到店消费（书店、餐饮等） | 24 |
| `ota` | 在线旅游（机票、酒店、景点） | 38 |
| 跨领域（如 `delivery,instore,ota`） | 多领域合并，用逗号分隔 | 66 |

## 任务配置字段（`workspace/vita.yaml`）

仅列出与任务运行直接相关的字段。

### task_config

- `dataset_path` (string，可选): 本地 JSON 或 JSONL 数据集文件路径，包含 VitaBench 任务数据。 与 `instance_id` 配合使用。
- `instance_id` (string，可选): 数据集文件中的任务实例 ID，格式为 `{domain}__{lang}-{task_id}`（如 `delivery__zh-10711001`）。
- `task_data` (dict，可选): 直接传入完整的任务 JSON 数据。 若提供此字段，则无需 `dataset_path` 和 `instance_id`。
- `language` (string，可选): 语言设置，可选 `chinese` 或 `english`。 若不指定，自动从 `instance_id` 推断（`zh` → 中文，`en` → 英文）。
- `evaluation_type` (string，默认 `trajectory`): 评估类型。 可选值：
  - `trajectory`: 滑动窗口 + Rubric 评估（默认，推荐）
  - `trajectory_full_traj_rubric`: 全轨迹 + Rubric 评估
  - `trajectory_sliding_wo_rubric`: 滑动窗口，不使用 Rubric
  - `trajectory_full_traj_wo_rubric`: 全轨迹，不使用 Rubric

> **注意**：`dataset_path` + `instance_id` 与 `task_data` 二选一，必须提供其中一种方式来指定任务数据。

### agent_config

- `max_steps` (int，默认 `300`): 最大模拟交互步数。
- `max_errors` (int，默认 `10`): 工具调用错误数超过此值时终止模拟。
- `enable_think` (bool，默认 `false`): 是否启用 agent LLM 的 thinking/reasoning 模式。
- `seed` (int，默认 `300`): 随机种子，用于结果复现。
- `user_llm_kwargs` (dict，可选): 传递给 user simulator LLM 的额外参数。
- `agent_llm_kwargs` (dict，可选): 传递给 agent LLM 的额外参数。

### openai_clients（三个角色）

`vita_task` 需要配置 **三个** OpenAI 兼容客户端，分别用于不同角色：

| 客户端名称 | 角色 | 是否必需 |
|-----------|------|---------|
| `main` | Agent LLM（执行任务的智能体） | 必需 |
| `user` | User Simulator LLM（模拟用户） | 可选，不配置时回退到 `main` |
| `evaluator` | Evaluator LLM（评估轨迹） | 可选，不配置时回退到 `main` |

> **建议**：为获得更准确的评测结果，推荐为三个角色分别配置不同的模型。 特别是 `evaluator` 建议使用推理能力强的模型。

## 执行流程

1. **加载任务数据**：从 `dataset_path` + `instance_id` 加载，或直接使用 `task_data`。
2. **推断参数**：从任务数据中提取 `domain`、`task_id`，从 `instance_id` 自动推断 `language`。
3. **构建环境**：根据 `domain` 构建对应的 `Environment`（包含工具集、数据库等），跨领域任务会自动合并多个领域的环境。
4. **准备系统提示词**：从环境策略（policy）中提取系统提示词，并填入当前时间与星期信息。
5. **运行 vita_agent**：启动三方协作循环：
   - **Agent**：调用 LLM 产生回复或工具调用。
   - **User Simulator**：模拟真实用户进行多轮对话。
   - **Environment**：执行工具调用并返回结果。
6. **评估轨迹**：使用滑动窗口方式将对话切分为多个窗口，由 Evaluator LLM 逐窗口对照 Rubric 评估。
7. **产出结果**：汇总评分、终止原因、Rubric 详情和模拟摘要。

## 最小运行示例

### 1) 使用本地数据集

编辑 `workspace/vita.yaml`：

```yaml
imports:
  - nexus.extensions.tasks.vita
  - nexus.extensions.openai_clients.litellm

task_name: "vita_task"
task_config:
  dataset_path: "workspace/vita/test.jsonl"
  instance_id: "delivery__zh-10711001"
  evaluation_type: "trajectory"

agent_name: "vita_agent"
agent_config:
  max_steps: 300
  max_errors: 10
  enable_think: false
  seed: 300

openai_clients:
  main:
    name: "litellm"
    config:
      client_args:
        api_key: "<your-api-key>"
        base_url: "https://api.example.com/v1"
      request_args:
        model: "openai/qwen3.5-plus"
```

执行任务：

```bash
uv run nexus run-single workspace/vita.yaml
```

### 2) 配置三个独立 LLM（推荐）

为三个角色配置不同模型以获得更好的评测效果：

```yaml
imports:
  - nexus.extensions.tasks.vita
  - nexus.extensions.openai_clients.litellm

task_name: "vita_task"
task_config:
  dataset_path: "workspace/vita/test.jsonl"
  instance_id: "delivery__zh-10711001"
  evaluation_type: "trajectory"

agent_name: "vita_agent"
agent_config:
  max_steps: 300
  max_errors: 10
  enable_think: false
  seed: 300

openai_clients:
  # Agent LLM：执行任务的智能体
  main:
    name: "litellm"
    config:
      client_args:
        api_key: "<your-api-key>"
        base_url: "https://api.example.com/v1"
      request_args:
        model: "openai/qwen3.5-plus"
  # User Simulator LLM：模拟真实用户
  user:
    name: "litellm"
    config:
      client_args:
        api_key: "<your-api-key>"
        base_url: "https://api.example.com/v1"
      request_args:
        model: "openai/kimi-k2.5"
  # Evaluator LLM：评估轨迹质量
  evaluator:
    name: "litellm"
    config:
      client_args:
        api_key: "<your-api-key>"
        base_url: "https://api.example.com/v1"
      request_args:
        model: "openai/qwen3-max"

processors:
  chat:
    name: "chat_tracker"
  trace:
    name: "json_line"
    args:
      file_path: "workspace/vita_trace.jsonl"
```

### 3) 使用原生 OpenAI 客户端

如果不使用 litellm，可改用内置的 `openai` 客户端：

```yaml
imports:
  - nexus.extensions.tasks.vita

task_name: "vita_task"
task_config:
  dataset_path: "workspace/vita/test.jsonl"
  instance_id: "ota__en-A0812005"
  language: "english"
  evaluation_type: "trajectory"

agent_name: "vita_agent"
agent_config:
  max_steps: 300
  enable_think: true

openai_clients:
  main:
    name: "openai"
    config:
      client_args:
        api_key: "<your-openai-api-key>"
        base_url: "https://api.openai.com/v1"
      request_args:
        model: "gpt-4.1"
```

## 任务输出

`VitaTaskOutput` 包含以下字段：

- `score` (float): 综合评分（0.0 ~ 1.0）。
- `termination_reason` (string): 模拟终止原因：
  - `agent_stop`: Agent 主动结束对话。
  - `user_stop`: User Simulator 主动结束对话。
  - `max_steps`: 达到最大交互步数。
  - `too_many_errors`: 工具调用错误次数超限。
- `rubric_score` (float): 首项 Rubric 评分。
- `reward_details` (dict): 详细评估信息，包含各 Rubric 的 `met`（是否达成）和 `justification`（评估理由）。
- `simulation_summary` (dict): 模拟摘要，包含 `task_id`、`termination_reason`、`duration`、`agent_cost`、`user_cost`、`num_messages` 等。

## 批量运行（Batch Mode）

对于大规模评测场景，可使用 `nexus run-batch` 配合 generator 文件批量运行多个 VitaBench 任务。 Generator 位于 `nexus/extensions/tasks/vita/generator.py`。

### 环境变量

Generator 通过环境变量配置运行参数：

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `VITA_DOMAIN` | 领域名称，支持单领域（`ota`/`delivery`/`instore`）和跨领域（逗号分隔，如 `delivery,instore,ota`） | `ota` |
| `VITA_LANGUAGE` | 语言：`chinese` 或 `english` | `chinese` |
| `VITA_DATA_DIR` | VitaBench 数据目录路径 | 自动检测 |
| `VITA_TASK_IDS` | 逗号分隔的任务 ID 列表，仅运行指定任务 | 空（运行全部） |
| `VITA_SKIP` | 跳过前 N 条任务 | `0` |
| `VITA_LIMIT` | 最多运行 N 条任务 | 无限制 |
| `VITA_MAX_STEPS` | 最大交互步数 | `300` |
| `VITA_ENABLE_THINK` | 启用 agent thinking 模式（`1`/`true`） | `false` |
| `VITA_SEED` | 随机种子 | `300` |
| `OPENAI_CLIENT_NAME` | Agent LLM 客户端类型 | `openai` |
| `OPENAI_API_KEY` | Agent LLM API 密钥 | `sk-placeholder` |
| `OPENAI_BASE_URL` | Agent LLM API 地址 | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | Agent LLM 模型名称 | `gpt-4.1` |
| `VITA_USER_API_KEY` | User Simulator LLM API 密钥（可选，不设置则回退到 Agent LLM） | 空 |
| `VITA_USER_BASE_URL` | User Simulator LLM API 地址 | 空 |
| `VITA_USER_MODEL` | User Simulator LLM 模型名称 | 空 |
| `VITA_EVAL_API_KEY` | Evaluator LLM API 密钥（可选，不设置则回退到 Agent LLM） | 空 |
| `VITA_EVAL_BASE_URL` | Evaluator LLM API 地址 | 空 |
| `VITA_EVAL_MODEL` | Evaluator LLM 模型名称 | 空 |

### 示例命令

**1) 单领域批量评测**

运行 OTA 领域全部任务：

```bash
VITA_DOMAIN=ota \
VITA_LANGUAGE=chinese \
OPENAI_API_KEY=sk-xxx \
OPENAI_BASE_URL=https://api.example.com/v1 \
OPENAI_MODEL=qwen3.5-plus \
uv run nexus run-batch \
  -g ./nexus/extensions/tasks/vita/generator.py \
  -d ./workspace/vita_batch_ota.db \
  --max-concurrent 20
```

**2) 跨领域批量评测**

运行跨领域（delivery + instore + ota）全部任务：

```bash
VITA_DOMAIN=delivery,instore,ota \
VITA_LANGUAGE=chinese \
OPENAI_API_KEY=sk-xxx \
OPENAI_BASE_URL=https://api.example.com/v1 \
OPENAI_MODEL=qwen3.5-plus \
uv run nexus run-batch \
  -g ./nexus/extensions/tasks/vita/generator.py \
  -d ./workspace/vita_batch_cross.db \
  --max-concurrent 10
```

**3) 使用独立 LLM 配置批量评测**

为三个角色分别配置不同的模型：

```bash
VITA_DOMAIN=delivery \
VITA_LANGUAGE=chinese \
OPENAI_API_KEY=sk-agent-key \
OPENAI_BASE_URL=https://api.example.com/v1 \
OPENAI_MODEL=qwen3.5-plus \
VITA_USER_API_KEY=sk-user-key \
VITA_USER_BASE_URL=https://api.example.com/v1 \
VITA_USER_MODEL=kimi-k2.5 \
VITA_EVAL_API_KEY=sk-eval-key \
VITA_EVAL_BASE_URL=https://api.example.com/v1 \
VITA_EVAL_MODEL=qwen3-max \
uv run nexus run-batch \
  -g ./nexus/extensions/tasks/vita/generator.py \
  -d ./workspace/vita_batch_delivery.db \
  --max-concurrent 20
```

**4) 英文任务评测**

```bash
VITA_DOMAIN=ota \
VITA_LANGUAGE=english \
OPENAI_API_KEY=sk-xxx \
OPENAI_BASE_URL=https://api.openai.com/v1 \
OPENAI_MODEL=gpt-4.1 \
uv run nexus run-batch \
  -g ./nexus/extensions/tasks/vita/generator.py \
  -d ./workspace/vita_batch_en.db \
  --max-concurrent 20
```

**5) 指定任务 ID 运行**

只运行特定的几个任务：

```bash
VITA_DOMAIN=delivery \
VITA_TASK_IDS=10711001,10711002,10711003 \
OPENAI_API_KEY=sk-xxx \
OPENAI_BASE_URL=https://api.example.com/v1 \
OPENAI_MODEL=qwen3.5-plus \
uv run nexus run-batch \
  -g ./nexus/extensions/tasks/vita/generator.py \
  -d ./workspace/vita_batch_specific.db \
  --max-concurrent 5
```

**6) 启用 Thinking 模式**

```bash
VITA_DOMAIN=ota \
VITA_ENABLE_THINK=true \
VITA_MAX_STEPS=200 \
OPENAI_API_KEY=sk-xxx \
OPENAI_BASE_URL=https://api.example.com/v1 \
OPENAI_MODEL=qwen3.5-plus \
uv run nexus run-batch \
  -g ./nexus/extensions/tasks/vita/generator.py \
  -d ./workspace/vita_batch_think.db \
  --max-concurrent 10
```

**7) 部分运行（跳过/限制）**

跳过前 50 条，只运行后续 20 条：

```bash
VITA_DOMAIN=ota \
VITA_SKIP=50 \
VITA_LIMIT=20 \
OPENAI_API_KEY=sk-xxx \
OPENAI_BASE_URL=https://api.example.com/v1 \
OPENAI_MODEL=qwen3.5-plus \
uv run nexus run-batch \
  -g ./nexus/extensions/tasks/vita/generator.py \
  -d ./workspace/vita_batch_partial.db \
  --max-concurrent 10
```

### 结果存储、恢复运行、重试失败任务

参见 [run-batch.md](run-batch.md)。

## 数据集格式

开源数据集参见：[VitaBench](https://git.woa.com/hunyuan-dataset/business-public/VitaBench)

VitaBench 任务数据为 JSON 格式，每条任务包含以下核心字段：

```json
{
  "instance_id": "delivery__zh-10711001",
  "id": "10711001",
  "domain": "delivery",
  "environment": {
    "...": "领域相关的环境数据（商户、商品、订单、用户位置等）"
  },
  "user_scenario": {
    "user_profile": "用户画像描述",
    "scenario": "场景描述"
  },
  "instructions": "用户的初始指令",
  "evaluation_criteria": {
    "overall_rubrics": ["评估 rubric 列表"],
    "expected_states": [
      {
        "state_rubrics": ["状态相关的 rubric"]
      }
    ]
  }
}
```

数据集可以是：
- **单个 JSON 文件**：包含单条任务或任务数组。
- **JSONL 文件**：每行一条 JSON 任务数据，每条需包含 `instance_id` 字段。

## 关于 VitaBench

VitaBench 是一个面向生活服务场景的 LLM Agent 评测基准，涵盖 66 种工具、100 个跨场景任务和 300 个单场景任务。 评测结果显示，即使是最先进的模型在跨场景任务上的成功率也仅为 32.5%。

更多信息请参阅：
- 论文：[VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks](https://arxiv.org/abs/2509.26490)
- 数据集：[HuggingFace](https://huggingface.co/datasets/meituan-longcat/VitaBench)
- 调研报告：[Vita Bench 生活AI Agent数据集与任务场景调研](https://iwiki.woa.com/p/4018768989)
