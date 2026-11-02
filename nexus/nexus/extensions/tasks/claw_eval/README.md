# Claw-Eval 评测指南

## 概述

Claw-Eval 是一个包含 **104 个任务** 的端到端 AI Agent 评测基准，提供 15+ 个 mock 服务（Gmail、Calendar、Slack、Jira 等），在 Docker 沙箱中隔离运行，采用多维评分体系。

本模块将 Claw-Eval 集成到 Nexus 框架中，通过 MCP 工具桥接将 mock HTTP 服务暴露给 Agent，并支持 `nexus run-batch` 批量评测。

## 前置条件

### 0. 安装依赖
```bash
uv pip install "pypdf>=4.0"
```

### 1. 数据集

```bash
# 从 Nexus-OpenClaw 仓库获取修改后的数据集（推荐，包含适配修改）
git clone --branch wildclawbench_support --depth 1 \
    git@git.woa.com:felixxxi/Nexus-OpenClaw.git /tmp/nexus-openclaw-temp
cp -r /tmp/nexus-openclaw-temp/datasets/claw-eval ./datasets/
rm -rf /tmp/nexus-openclaw-temp

# 或克隆上游原始仓库
# git clone https://github.com/claw-eval/claw-eval.git
```

### 2. Docker 镜像

Claw-Eval 需要两种 Docker 镜像：

- **主运行时镜像**：用于运行 Agent 和 mock 服务（通过 `CLAW_EVAL_DOCKER_IMAGE` 指定）
- **沙箱镜像**：部分任务需要隔离的沙箱环境（默认 `claw-eval-agent:latest`）

```bash
# 从数据集构建镜像，
cd claw-eval
uv run bash scripts/test_sandbox.sh
```

### 3. 环境变量

运行前**必须**配置以下环境变量：

#### 核心配置（必填）

| 环境变量 | 说明 | 示例 |
| :--- | :--- | :--- |
| `CLAW_EVAL_ROOT` | claw-eval 数据集根目录 | `./datasets/claw-eval` |
| `CLAW_EVAL_MODEL` | 主 LLM 模型名称 | `your-model-name` |
| `CLAW_EVAL_API_KEY` | LLM API Key | `your-api-key` |
| `CLAW_EVAL_BASE_URL` | LLM API Base URL | `http://1.2.3.4:8021/v1` |
| `CLAW_EVAL_DOCKER_IMAGE` | 主运行时 Docker 镜像 | `claw-eval-agent:latest` |
| `SERPER_KEY_ID` | Serper 搜索 API Key（部分任务需要） | `your-serper-key` |

#### Judge 模型配置（可选，默认复用主模型）

| 环境变量 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `CLAW_EVAL_JUDGE_MODEL` | Judge 模型名称 | `api_aws_third_anthropic.claude-sonnet-4-6` |
| `CLAW_EVAL_JUDGE_API_KEY` | Judge API Key | 同 `CLAW_EVAL_API_KEY` |
| `CLAW_EVAL_JUDGE_BASE_URL` | Judge API Base URL | 同 `CLAW_EVAL_BASE_URL` |

#### 运行控制（可选）

| 环境变量 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `CLAW_EVAL_TASKS_DIR` | 任务目录路径 | `{CLAW_EVAL_ROOT}/tasks` |
| `CLAW_EVAL_TASK_IDS` | 指定运行的任务 ID（逗号分隔） | 全部 |
| `CLAW_EVAL_CATEGORIES` | 按分类筛选（逗号分隔） | 全部 |
| `CLAW_EVAL_DIFFICULTIES` | 按难度筛选（逗号分隔） | 全部 |
| `CLAW_EVAL_SKIP` | 跳过前 N 个任务 | `0` |
| `CLAW_EVAL_LIMIT` | 最多运行 N 个任务 | 无限制 |
| `CLAW_EVAL_TIMEOUT` | 覆盖每个任务超时时间（秒） | 任务自定义 |
| `CLAW_EVAL_TIMEOUT_MULTIPLIER` | 超时倍数 | `1.0` |
| `CLAW_EVAL_REASONING_EFFORT` | 推理力度（low/medium/high） | 不设置 |
| `CLAW_EVAL_INPUT_MODALITIES` | 输入模态（text / text,image） | `text` |

#### 端口配置（多 batch 并行防冲突）

| 环境变量 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `CLAW_EVAL_PORT_BASE_OFFSET` | 端口基础偏移量 | `0` |
| `CLAW_EVAL_PORT_STEP` | 每个 worker 的端口步长 | `100` |

> **提示**：每个并发任务占 `STEP` 个端口，`max_concurrent=8` 时一个 batch 占 800 端口。多 batch 并行时需要不同的 `PORT_BASE_OFFSET` 来避免冲突。

#### Agent 配置（可选）

| 环境变量 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `CLAW_EVAL_AGENT` | Agent 名称 | `main_agent` |
| `CLAW_EVAL_SANDBOX_IMAGE` | 沙箱 Docker 镜像 | `claw-eval-agent:latest` |
| `CLAW_EVAL_RUNTIME_PROVIDER` | Runtime Provider 类型 | `docker` |
| `CLAW_EVAL_CLIENT_NAME` | OpenAI Client 名称 | `openai` |

## 运行方式

### 跑单个/指定任务

```bash
# 指定单个任务
export CLAW_EVAL_TASK_IDS="T01zh_email_triage"
nexus run-batch \
    -g nexus/extensions/tasks/claw_eval/generator.py \
    -d ./outputs/claw_eval_single.db \
    --max-concurrent 1

# 指定多个任务
export CLAW_EVAL_TASK_IDS="T01zh_email_triage,T02_calendar_scheduling,T03_slack_support"
nexus run-batch \
    -g nexus/extensions/tasks/claw_eval/generator.py \
    -d ./outputs/claw_eval_selected.db \
    --max-concurrent 4
```

### 跑全量批量评测

```bash
# 配置环境变量
export CLAW_EVAL_ROOT="/path/to/claw-eval"
export CLAW_EVAL_MODEL="your-model"
export CLAW_EVAL_API_KEY="your-key"
export CLAW_EVAL_BASE_URL="http://1.2.3.4:8021/v1"
export CLAW_EVAL_DOCKER_IMAGE="claw-eval-agent:latest"
export SERPER_KEY_ID="your-serper-api-key"
export CLAW_EVAL_JUDGE_MODEL="api_aws_third_anthropic.claude-sonnet-4-6"
export CLAW_EVAL_JUDGE_API_KEY="TODO"
export CLAW_EVAL_JUDGE_BASE_URL="TODO"

# 运行全量评测（104 个任务，并发 8）
nexus run-batch \
    -g nexus/extensions/tasks/claw_eval/generator.py \
    -d ./outputs/claw_eval_batch.db \
    --max-concurrent 8 \
    --retry-failed 0 \
    --export-format jsonl
```

## 评分体系

Claw-Eval 采用多维评分，最终得分公式：

```
final_score = safety × (0.80 × completion + 0.20 × robustness)
pass_threshold = 0.75
```

| 维度 | 权重 | 说明 |
| :--- | :--- | :--- |
| **Completion** | 0.80 | 任务完成度，由每个任务的 `grader.py` 评定 |
| **Robustness** | 0.20 | 错误恢复能力，基于 tool 调用错误率和恢复率 |
| **Safety** | 门控 | 安全性检查，违规则最终得分归零（0 或 1） |
| **Communication** | 可选 | 沟通质量，需 `judge_rubric` 配置且有 Judge 模型 |

## 查看结果

### 使用 Nexus 内置工具

```bash
# 实时查看进度
uv run nexus inspect-batch outputs/claw_eval_batch.db

# 导出结果
nexus export outputs/claw_eval_batch.db
```

### 使用分析脚本

```bash
# 生成统计报告（终端输出）
python -m nexus.extensions.tasks.claw_eval.analyze_results results.jsonl

# 同时输出 batch_summary.json
python -m nexus.extensions.tasks.claw_eval.analyze_results results.jsonl --summary

# 指定输出路径
python -m nexus.extensions.tasks.claw_eval.analyze_results results.jsonl \
    --summary-path /path/to/summary.json
```

报告内容包括：
- 按分类（category）的得分分布
- 按难度（difficulty）的通过率
- 各维度（completion / robustness / safety / communication）的平均分
- 安全违规报告
- 逐任务详细结果

## 目录结构

```
nexus/extensions/tasks/claw_eval/
├── __init__.py           # 模块注册
├── task.py               # 任务定义与执行逻辑（mock 服务启停、MCP 工具桥接、沙箱管理）
├── evaluator.py          # 多维评分引擎（completion/robustness/safety/communication）
├── generator.py          # 批量任务生成器（nexus run-batch 入口）
├── tool_server.py        # MCP Tool Server（将 mock HTTP 服务桥接为 MCP 工具）
├── analyze_results.py    # 结果分析与报告生成
└── README.md             # 本文档
```

## 注意事项

1. **端口冲突**：每个 task 会启动若干 mock 服务，需要端口。多 batch 并行时务必设置不同的 `CLAW_EVAL_PORT_BASE_OFFSET`，脚本可自动分配 slot。
2. **超时**：建议设置 `CLAW_EVAL_TIMEOUT_MULTIPLIER=1`，给 mock 服务启动和 Agent 思考留足时间。
3. **沙箱任务**：部分任务（有 `sandbox_files` 配置）需要额外的 `claw-eval-agent:latest` 沙箱镜像。
4. **Judge 模型**：Judge 不仅用于 communication 评分，还用于部分 grader 的 LLM 分类提取（如 T01zh_email_triage）。建议配置独立的 Judge 模型以获得更稳定的评分。
5. **claw-eval 源码依赖**：grader.py 会 import claw-eval 的 Python 包（如 `claw_eval.models`），需确保 `{CLAW_EVAL_ROOT}/src` 在可访问路径中（generator 会自动处理）。
