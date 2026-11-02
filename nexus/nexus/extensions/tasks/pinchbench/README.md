# PinchBench 评测指南

## 概述

PinchBench 是一个包含 **23 个真实场景任务** 的 OpenClaw Agent 评测基准。任务以 Markdown + YAML frontmatter 格式定义，支持三种评分方式：自动化评分（`automated`）、LLM 裁判评分（`llm_judge`）和混合评分（`hybrid`）。

本模块将 PinchBench 集成到 Nexus 框架中，在 Docker 容器内运行 OpenClaw Agent 执行任务，并在容器内完成自动评分。

## 前置条件

### 1. 数据集

```bash
# 从 Nexus-OpenClaw 仓库获取修改后的数据集（推荐，包含适配修改）
git clone --branch wildclawbench_support --depth 1 \
    git@git.woa.com:felixxxi/Nexus-OpenClaw.git /tmp/nexus-openclaw-temp
cp -r /tmp/nexus-openclaw-temp/datasets/pinchbench ./datasets/
rm -rf /tmp/nexus-openclaw-temp

# 或克隆上游原始仓库
# git clone https://github.com/pinchbench/skill.git
```

数据集目录结构：
```
pinchbench/
├── tasks/                  # 23 个任务文件（task_*.md）
│   ├── task_01_django_debug.md
│   ├── task_02_react_refactor.md
│   └── ...
├── assets/                 # 任务所需的附件资源
│   ├── task_01/
│   └── ...
└── scripts/                # 辅助脚本
```

### 2. Docker 镜像

PinchBench 使用 Docker 容器运行 OpenClaw Agent：

```bash
# 拉取预构建镜像
docker pull mirrors.tencent.com/hunyuan_yanguan/sweb.eval.x86_64.django__django-12155
```

### 3. 环境变量

运行前**必须**配置以下环境变量：

#### 核心配置（必填）

| 环境变量 | 说明 | 示例 |
| :--- | :--- | :--- |
| `OPENCLAW_MODEL` | OpenClaw Agent 模型名称 | `your-model-name` |
| `OPENCLAW_LITELLM_API_KEY` | LLM API Key | `your-api-key` |
| `OPENCLAW_LITELLM_BASE_URL` | LLM API Base URL | `http://1.2.3.4:8021/v1` |
| `PINCHBENCH_IMAGE` | Docker 镜像名称 | `mirrors.tencent.com/hunyuan_yanguan/sweb.eval.x86_64.django__django-12155` |

#### 任务路径配置

| 环境变量 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `PINCHBENCH_TASKS_DIR` | 任务文件目录 | `datasets/pinchbench/tasks` |
| `PINCHBENCH_ASSETS_DIR` | 资产文件目录 | `datasets/pinchbench/assets` |

#### Judge 模型配置（可选，`llm_judge` / `hybrid` 类型任务需要）

| 环境变量 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `JUDGE_MODEL` | Judge 模型名称 | `api_aws_third_anthropic.claude-sonnet-4-6` |
| `JUDGE_API_KEY` | Judge API Key | 同 `OPENCLAW_LITELLM_API_KEY` |
| `JUDGE_BASE_URL` | Judge API Base URL | 同 `OPENCLAW_LITELLM_BASE_URL` |

#### OpenClaw Agent 配置（可选）

| 环境变量 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `OPENCLAW_VERSION` | OpenClaw Agent 版本 | `2026.3.13` |
| `OPENCLAW_NPM_REGISTRY` | NPM 镜像源 | 空（默认源） |
| `OPENCLAW_CONTEXT_WINDOW` | 上下文窗口大小 | `200000` |
| `OPENCLAW_MAX_TOKENS` | 最大输出 token 数 | `65536` |

#### 运行控制（可选）

| 环境变量 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `PINCHBENCH_TASK_IDS` | 指定运行的任务 ID（逗号分隔） | 全部 |
| `PINCHBENCH_SKIP` | 跳过前 N 个任务 | `0` |
| `PINCHBENCH_LIMIT` | 最多运行 N 个任务 | 无限制 |
| `PINCHBENCH_AUTOMATED_ONLY` | 仅运行自动评分任务 | `false` |
| `PINCHBENCH_TIMEOUT` | 覆盖每个任务超时时间（秒） | 任务 frontmatter 定义 |
| `PINCHBENCH_TIMEOUT_MULTIPLIER` | 超时倍数 | `1.0` |
| `PINCHBENCH_REASONING_EFFORT` | 推理力度（low/medium/high/none） | 不设置 |
| `PINCHBENCH_FORCE_REBUILD` | 是否强制重建 Docker 镜像 | `false` |

## 运行方式

### 跑单个/指定任务

```bash
# 指定单个任务（可以用文件名 stem 或 frontmatter 中的 id）
export PINCHBENCH_TASK_IDS="task_01_django_debug"
nexus run-batch \
    -g nexus/extensions/tasks/pinchbench/generator.py \
    -d ./outputs/pinchbench_single.db \
    --max-concurrent 1

# 指定多个任务
export PINCHBENCH_TASK_IDS="task_01_django_debug,task_02_react_refactor"
nexus run-batch \
    -g nexus/extensions/tasks/pinchbench/generator.py \
    -d ./outputs/pinchbench_selected.db \
    --max-concurrent 4
```

### 跑全量批量评测

```bash
# 配置核心环境变量
export OPENCLAW_MODEL="your-model"
export OPENCLAW_LITELLM_API_KEY="your-key"
export OPENCLAW_LITELLM_BASE_URL="http://1.2.3.4:8021/v1"
export PINCHBENCH_IMAGE="mirrors.tencent.com/hunyuan_yanguan/sweb.eval.x86_64.django__django-12155"
export PINCHBENCH_TASKS_DIR="/path/to/pinchbench/tasks"
export PINCHBENCH_ASSETS_DIR="/path/to/pinchbench/assets"

# 配置 Judge（llm_judge / hybrid 任务需要）
export JUDGE_MODEL="api_aws_third_anthropic.claude-sonnet-4-6"
export JUDGE_API_KEY="your-judge-key"
export JUDGE_BASE_URL="http://llm-api.model-eval.woa.com/v1"

# 运行全量评测（23 个任务，并发 4）
nexus run-batch \
    -g nexus/extensions/tasks/pinchbench/generator.py \
    -d ./outputs/pinchbench_batch.db \
    --max-concurrent 4 \
    --retry-failed 1 \
    --export-format jsonl
```

### 仅运行自动评分任务

```bash
export PINCHBENCH_AUTOMATED_ONLY=true
nexus run-batch \
    -g nexus/extensions/tasks/pinchbench/generator.py \
    -d ./outputs/pinchbench_auto.db \
    --max-concurrent 4
```


## 评分体系

PinchBench 支持三种评分方式，由每个任务的 YAML frontmatter 中 `grading_type` 字段指定：

### 1. 自动评分（`automated`）

任务 Markdown 中嵌入 Python 评分代码，包含 `grade(transcript, workspace_path)` 函数。评分代码在 Docker 容器内执行，可以访问 Agent 产生的文件和工作目录。

返回值为 `dict[str, float]`，最终得分取所有指标的平均值。

### 2. LLM 裁判评分（`llm_judge`）

使用 Judge 模型根据任务的 `expected_behavior`、`grading_criteria` 和可选的 `llm_judge_rubric` 进行评分。

评分标准：
- 1.0 = 卓越表现
- 0.6-0.7 = 一般可接受的完成度
- 对不必要的步骤、冗余输出、低效工具使用扣分

### 3. 混合评分（`hybrid`）

组合自动评分和 LLM 裁判评分，权重由任务的 `hybrid_weights` 字段配置（默认各 50%）：

```
hybrid_score = (auto_score × auto_weight + llm_score × llm_weight) / total_weight
```

## 查看结果

### 使用 Nexus 内置工具

```bash
# 实时查看进度
uv run nexus inspect-batch outputs/pinchbench_batch.db

# 导出结果
nexus export outputs/pinchbench_batch.db
```

### 使用分析脚本

```bash
# 生成统计报告（终端输出）
python -m nexus.extensions.tasks.pinchbench.analyze_results \
    --results_file ./outputs/results.jsonl
```

报告内容包括：
- 按分类（category）的得分分布
- 按评分类型（automated / llm_judge / hybrid）的统计
- 逐任务详细结果与耗时
- 整体通过率和平均分

## 目录结构

```
nexus/extensions/tasks/pinchbench/
├── __init__.py           # 模块注册
├── task.py               # 任务定义与执行逻辑（OpenClaw Agent 运行、文件准备）
├── evaluator.py          # 评分引擎（automated / llm_judge / hybrid）
├── generator.py          # 批量任务生成器（nexus run-batch 入口）
├── analyze_results.py    # 结果分析与报告生成
└── README.md             # 本文档
```

## 注意事项

1. **超时配置**：建议设置 `PINCHBENCH_TIMEOUT_MULTIPLIER=10`，因为 OpenClaw Agent 需要 npm install、onboard 等初始化步骤。
2. **NPM 镜像**：内网环境建议设置 `OPENCLAW_NPM_REGISTRY=http://mirrors.tencent.com/npm/`，加速 OpenClaw Agent 安装。
3. **资产文件**：部分任务需要 `assets/` 目录中的文件（frontmatter 中有 `workspace_files` 且包含 `source` 字段），确保 `PINCHBENCH_ASSETS_DIR` 指向正确路径。
4. **Judge 模型**：`llm_judge` 和 `hybrid` 类型的任务需要配置 Judge 模型。如果只想快速验证，可以设置 `PINCHBENCH_AUTOMATED_ONLY=true` 仅运行自动评分任务。
5. **容器内评分**：自动评分代码在 Docker 容器内执行（非宿主机），可以访问 Agent 在 `/workspace` 中产生的所有文件。
