# WildClawBench 评测指南

## 概述

WildClawBench 是一个包含 **60 个任务** 的端到端 OpenClaw Agent 评测基准，涵盖 6 大类别、25 个多模态任务。Agent 在 Docker 容器内运行，任务完成后在容器内执行自动评分。

### 任务分类

| 类别 | 说明 | 任务数 |
| :--- | :--- | :--- |
| `01_Productivity_Flow` | 生产力工作流 | ~10 |
| `02_Code_Intelligence` | 代码智能 | ~12 |
| `03_Social_Interaction` | 社交交互 | ~10 |
| `04_Search_Retrieval` | 搜索与检索 | ~11 |
| `05_Creative_Synthesis` | 创意合成 | ~11 |
| `06_Safety_Alignment` | 安全对齐 | ~6 |

其中 25 个任务需要多模态（图像/视频理解）能力，可通过 `WILDCLAW_MODALITY` 环境变量进行筛选。

## 前置条件

### 1. 数据集

```bash
# 从 Nexus-OpenClaw 仓库获取源代码
git clone --branch wildclawbench_support --depth 1 \
    git@git.woa.com:felixxxi/Nexus-OpenClaw.git /tmp/nexus-openclaw-temp
cp -r /tmp/nexus-openclaw-temp/datasets/WildClawBench ./datasets/
rm -rf /tmp/nexus-openclaw-temp

```

数据集目录结构：
```
WildClawBench/
├── tasks/                          # 60 个任务
│   ├── 01_Productivity_Flow/       # 按类别组织
│   │   ├── 01_Productivity_Flow_task_1_xxx.md
│   │   └── ...
│   ├── 02_Code_Intelligence/
│   ├── 03_Social_Interaction/
│   ├── 04_Search_Retrieval/
│   ├── 05_Creative_Synthesis/
│   └── 06_Safety_Alignment/
├── skills/                         # OpenClaw skills 定义
├── eval/                           # 评分脚本
├── docker/                         # Docker 镜像构建
└── prepare.sh                      # 数据准备脚本
```

### 2. 数据准备

部分任务需要预先准备 workspace 数据：

```bash
cd WildClawBench
bash prepare.sh
```

`prepare.sh` 会下载任务所需的代码仓库、数据文件等到各任务的 workspace 目录中。

### 3. Docker 镜像

WildClawBench 使用预构建的 Docker 镜像运行 Agent：

```bash
# 从镜像文件加载（推荐）
docker load -i wildclawbench-ubuntu-v1.2.tar

# 或从 Dockerfile 构建
cd WildClawBench/docker
docker build -t wildclawbench-ubuntu:v1.2 .
```

### 4. 环境变量

运行前**必须**配置以下环境变量：

#### 核心配置（必填）

| 环境变量 | 说明 | 示例 |
| :--- | :--- | :--- |
| `WILDCLAW_ROOT` | WildClawBench 数据集根目录 | `/path/to/WildClawBench` |
| `WILDCLAW_IMAGE` | Docker 镜像名称 | `wildclawbench-ubuntu:v1.2` |
| `OPENCLAW_MODEL` | OpenClaw Agent 模型名称 | `your-model-name` |
| `OPENCLAW_LITELLM_API_KEY` | LLM API Key | `your-api-key` |
| `OPENCLAW_LITELLM_BASE_URL` | LLM API Base URL | `http://1.2.3.4:8021/v1` |
| `SERPER_API_KEY` | Serper public API Key，或内部 gateway 的 `APP_ID:APP_KEY`（Search & Retrieval 任务必需） | `your-serper-key` |

#### Grading / Judge 配置（评分必需）

WildClawBench 的 Automated Checks 中部分评分函数使用 LLM 进行评判：

| 环境变量 | 说明 | 示例 |
| :--- | :--- | :--- |
| `OPENROUTER_API_KEY` | 评分 LLM API Key | `your-key` |
| `OPENROUTER_BASE_URL` | 评分 LLM Base URL | `http://llm-api.model-eval.woa.com/v1` |
| `OPENROUTER_BASE_URL_COMP` | 评分 LLM Completions URL | `http://llm-api.model-eval.woa.com/v1/chat/completions` |
| `GRADING_MODEL` | 评分使用的模型 | `api_aws_third_anthropic.claude-sonnet-4-6` |
| `GEMINI_API_KEY` | Gemini API Key（部分任务使用） | `your-gemini-key` |

#### OpenClaw Agent 配置（可选）

| 环境变量 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `OPENCLAW_VERSION` | OpenClaw Agent 版本 | `2026.3.11` |
| `OPENCLAW_NPM_REGISTRY` | NPM 镜像源 | 空（默认源） |
| `OPENCLAW_CONTEXT_WINDOW` | 上下文窗口大小 | `200000` |
| `OPENCLAW_MAX_TOKENS` | 最大输出 token 数 | `65536` |
| `OPENCLAW_SUPPORTS_IMAGES` | 模型是否支持图片输入 | `false` |

#### 运行控制（可选）

| 环境变量 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `WILDCLAW_CATEGORY` | 按类别筛选（逗号分隔，或 `all`） | `all` |
| `WILDCLAW_MODALITY` | 模态筛选：`all` / `text_only` / `multimodal` | `all` |
| `WILDCLAW_TASK_IDS` | 指定运行的任务 ID（逗号分隔） | 全部 |
| `WILDCLAW_SKIP` | 跳过前 N 个任务 | `0` |
| `WILDCLAW_LIMIT` | 最多运行 N 个任务 | 无限制 |
| `WILDCLAW_TIMEOUT` | 覆盖每个任务超时时间（秒） | 任务 frontmatter 定义 |
| `WILDCLAW_TIMEOUT_MULTIPLIER` | 超时倍数 | `1.0` |
| `WILDCLAW_FORCE_REBUILD` | 是否强制重建 Docker 镜像 | `false` |
| `WILDCLAW_REASONING_EFFORT` | 推理力度（low/medium/high/none/no_think） | 不设置 |
| `WILDCLAW_REQUEST_TIMEOUT` | HTTP 请求超时时间（秒） | `timeout + 900` |

#### 搜索 API 配置（可选）

| 环境变量 | 说明 |
| :--- | :--- |
| `SERPER_API_KEY` | Serper public API Key；`SERPER_API_TYPE=distillation` 时可写成 `APP_ID:APP_KEY` |
| `SERPER_API_TYPE` | `distillation` 时走内部 Serper-compatible gateway；不设置时走 public Serper |
| `SEARCH_SERPER_URL` | 内部 Serper-compatible gateway URL；`SERPER_API_TYPE=distillation` 时必须显式设置 |
| `SEARCH_SERPER_APP_ID` | 内部 gateway app id；未设置 `SERPER_API_KEY` 时会与 `SEARCH_SERPER_APP_KEY` 组合 |
| `SEARCH_SERPER_APP_KEY` | 内部 gateway app key |
| `SERPER_KEY_ID` | `SEARCH_SERPER_APP_ID` 的兼容别名 |

Nexus runtime 会启动本地 `127.0.0.1` SerperBridge，并把 OpenClaw 的 Brave provider 指向该 bridge。
新版 OpenClaw 还要求 `brave` plugin；Nexus 会先用配置的 NPM registry（默认
`http://mirrors.tencent.com/npm/`）尝试安装并启用 `@openclaw/brave-plugin`，再用
`openclaw infer web providers --json` 做 preflight，未通过时禁用 `web_search`。
OpenClaw 子进程只会收到占位 `BRAVE_API_KEY=nexus-serper-bridge`，不会收到真实 Serper 密钥或内部
gateway URL / app key。

## 运行方式

### 跑单个/指定任务

```bash
# 指定单个任务
export WILDCLAW_TASK_IDS="04_Search_Retrieval_task_11_fuzzy_repo_search"
nexus run-batch \
    -g nexus/extensions/tasks/wildclaw_bench/generator.py \
    -d ./outputs/wildclaw_single.db \
    --max-concurrent 1

# 指定多个任务
export WILDCLAW_TASK_IDS="01_Productivity_Flow_task_1_xxx,02_Code_Intelligence_task_1_xxx"
nexus run-batch \
    -g nexus/extensions/tasks/wildclaw_bench/generator.py \
    -d ./outputs/wildclaw_selected.db \
    --max-concurrent 4
```

### 按类别运行

```bash
# 只跑生产力工作流
export WILDCLAW_CATEGORY="01_Productivity_Flow"
nexus run-batch \
    -g nexus/extensions/tasks/wildclaw_bench/generator.py \
    -d ./outputs/wildclaw_productivity.db \
    --max-concurrent 4

# 只跑安全对齐
export WILDCLAW_CATEGORY="06_Safety_Alignment"
nexus run-batch \
    -g nexus/extensions/tasks/wildclaw_bench/generator.py \
    -d ./outputs/wildclaw_safety.db \
    --max-concurrent 4
```

### 按模态筛选

```bash
# 仅纯文本任务（35 个，不需要多模态能力）
export WILDCLAW_MODALITY="text_only"
nexus run-batch \
    -g nexus/extensions/tasks/wildclaw_bench/generator.py \
    -d ./outputs/wildclaw_text.db \
    --max-concurrent 4

# 仅多模态任务（25 个，需要图像/视频理解能力）
export WILDCLAW_MODALITY="multimodal"
export OPENCLAW_SUPPORTS_IMAGES="true"
nexus run-batch \
    -g nexus/extensions/tasks/wildclaw_bench/generator.py \
    -d ./outputs/wildclaw_multimodal.db \
    --max-concurrent 4
```

### 跑全量批量评测

```bash
# 配置核心环境变量
export WILDCLAW_ROOT="/path/to/WildClawBench"
export WILDCLAW_IMAGE="wildclawbench-ubuntu:v1.2"
export OPENCLAW_MODEL="your-model"
export OPENCLAW_LITELLM_API_KEY="your-key"
export OPENCLAW_LITELLM_BASE_URL="http://1.2.3.4:8021/v1"
export SERPER_API_KEY="your-serper-key"
# 内部 gateway 场景：
# export SERPER_API_TYPE="distillation"
# export SEARCH_SERPER_URL="http://your-serper-gateway/search"
# export SEARCH_SERPER_APP_ID="your-app-id"
# export SEARCH_SERPER_APP_KEY="your-app-key"

# 配置评分
export OPENROUTER_API_KEY="your-key"
export OPENROUTER_BASE_URL="http://llm-api.model-eval.woa.com/v1"
export OPENROUTER_BASE_URL_COMP="http://llm-api.model-eval.woa.com/v1/chat/completions"
export GRADING_MODEL="api_aws_third_anthropic.claude-sonnet-4-6"
export GEMINI_API_KEY="your-gemini-key"

# 运行全量评测（60 个任务，并发 4）
nexus run-batch \
    -g nexus/extensions/tasks/wildclaw_bench/generator.py \
    -d ./outputs/wildclaw_batch.db \
    --max-concurrent 4 \
    --retry-failed 1 \
    --export-format jsonl
```

## 评分体系

WildClawBench 使用容器内自动评分，每个任务的评分逻辑定义在任务 Markdown 的 `## Automated Checks` 节中，包含 `def grade(**kwargs)` 函数。

### 评分流程

1. Agent 在容器内执行任务
2. 评分脚本上传到**同一个容器**中执行
3. `grade()` 函数返回 `dict[str, float]`，包含各评分维度
4. `overall_score` 字段作为主得分；如无该字段，取所有数值字段的平均值

### 评分环境

评分代码在 Agent 运行完成后的同一容器内执行，可以：
- 读取 Agent 生成的文件（`/tmp_workspace/` 目录）
- 调用容器内的命令（如 `python3`, `node` 等）
- 使用 `GRADING_MODEL` 进行 LLM 评判（部分任务）
- 访问 Agent 的对话历史（通过 `transcript` 参数）

## 查看结果

### 使用 Nexus 内置工具

```bash
# 实时查看进度
uv run nexus inspect-batch outputs/wildclaw_batch.db

# 导出结果
nexus export outputs/wildclaw_batch.db
```

### 使用分析脚本

```bash
# 生成统计报告（终端输出）
python -m nexus.extensions.tasks.wildclaw_bench.analyze_results \
    --results_file ./outputs/results.jsonl
```

报告内容包括：
- 按类别（category）的得分分布
- 逐任务详细得分（含各评分子维度）
- 整体通过率和平均分
- 多模态 vs 纯文本任务的对比统计

## 多模态任务列表

以下 25 个任务需要图像/视频理解能力（需设置 `OPENCLAW_SUPPORTS_IMAGES=true`）：

| 类别 | 任务 |
| :--- | :--- |
| Productivity Flow | task_3_bibtex, task_8_real_image_category |
| Code Intelligence | task_3 ~ task_12（拼图、OCR、主页生成等） |
| Search Retrieval | task_7_location_search, task_9_artwork_search |
| Creative Synthesis | task_1 ~ task_11（赛事报告、海报、视频笔记等） |

## 目录结构

```
nexus/extensions/tasks/wildclaw_bench/
├── __init__.py           # 模块注册
├── task.py               # 任务定义与执行逻辑（OpenClaw Agent 运行、warmup、skill 安装）
├── evaluator.py          # 容器内自动评分引擎
├── generator.py          # 批量任务生成器（nexus run-batch 入口）
├── analyze_results.py    # 结果分析与报告生成
└── README.md             # 本文档
```

## 注意事项

1. **数据准备**：首次运行前务必执行 `prepare.sh`，部分任务需要下载代码仓库和数据文件到 workspace 目录。
2. **Docker 镜像代理**：`wildclawbench-ubuntu` 镜像可能内置了 HTTP 代理配置。Nexus 会自动清除容器内的代理环境变量（`http_proxy`, `https_proxy` 等），避免与本地网络冲突。
3. **容器卷挂载**：每个任务会将 `workspace_path/exec` 目录以只读方式挂载到容器的 `/app`，匹配原始 WildClawBench 的行为。
4. **NPM 镜像**：内网环境建议设置 `OPENCLAW_NPM_REGISTRY=http://mirrors.tencent.com/npm/`，加速 OpenClaw Agent 和新版 `brave` plugin 安装。
5. **搜索 API**：Search & Retrieval 类任务需要搜索 API。设置 Serper 环境变量后，Nexus runtime 会启动本地 SerperBridge，对 OpenClaw 伪装成 Brave Search API；新版 OpenClaw 通过 Brave provider `baseUrl` 配置接入，并会尝试安装/启用 `@openclaw/brave-plugin` 后做 preflight；旧版 OpenClaw 只 patch Brave endpoint 常量。
6. **评分超时**：部分任务的评分函数会调用外部 LLM（如 video_notes），需要较长时间。评分脚本超时设为 300 秒。
7. **请求超时**：HTTP 请求超时默认为 `agent_timeout + 900s`，为 OpenClaw 的 npm install、onboard、LLM warm-up、评分等预留足够缓冲。可通过 `WILDCLAW_REQUEST_TIMEOUT` 自定义。
