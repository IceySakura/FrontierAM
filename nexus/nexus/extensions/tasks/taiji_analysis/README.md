# Taiji Replay — Nexus Benchmark

将太极实例分析接入 Nexus 评估框架，评估 Agent 的诊断分析能力。

---

## 分支结构

本分支 `debug_agent_taiji_replay` 同时服务两个用途：

```
debug_agent_taiji_replay (nexus-agent/nexus.git)
├── nexus/                                  # Nexus 框架
│   └── extensions/tasks/
│       └── taiji_analysis/                 # Benchmark 代码（本 README 所在目录）
│           ├── __init__.py
│           ├── README.md                   # ← 本文件
│           ├── generator.py                # Generator：数据 → BatchItem
│           ├── task.py                     # Task：编排逻辑
│           ├── evaluator.py                # Evaluator：LLM Judge
│           ├── prompt.py                   # Judge prompt
│           ├── compute_metrics.py          # 指标统计
│           ├── data_prep/                  # 数据预处理（跑 nexus 前）
│           │   ├── collect_episodes.py     # 从 agentd DB 提取数据集
│           │   └── convert_dataset.py      # 数据转换（--mode all/rule/model）
│           ├── gt_management/              # GT 更换与检查
│           │   ├── replace_gt.py           # 用跑出的 prediction 替换 GT
│           │   ├── rejudge_with_new_gt.py  # 换 GT 重新 Judge（不重跑 Agent）
│           │   └── check_gt_source.py      # 检查结果 DB 的 GT 来源
│           ├── eval_frontend/              # 结果可视化 + 标注工具
│           │   ├── app.py                  # Web 前端服务
│           │   ├── filter_by_annotation.py # 按标注过滤数据集
│           │   └── backfill_judge_reasoning.py # 回填 judge reasoning
│           ├── taiji_analysis_judge/       # Judge 准确性评估
│           │   ├── README.md               # 说明文档
│           │   ├── build_judge_benchmark.py # 构建 Judge benchmark
│           │   ├── eval_judge_benchmark.py  # 评估 Judge 准确性
│           │   └── run_judge_benchmark.py   # 运行 Judge benchmark
│           ├── add_cases/                  # 新增 case 流程
│           │   ├── README.md               # 完整流程说明
│           │   └── init_cases.py           # 初始化 case → dataset
│           ├── CLAUDE.md                   # 容器工作区 — Agent 行为指令
│           ├── episodes/                   # 评测数据（collect_episodes.py 产出）
│           ├── DESIGN.md                   # 详细设计文档
│           └── QUICKSTART.md               # 快速运行指南
├── pyproject.toml
└── ...
```

**调度节点**：clone 整个分支，安装 nexus 包，执行 `nexus run-batch`。

**容器内**：`pre_start_scripts` clone 同一分支，只取 `.agent/` 和 `CLAUDE.md`。

---

## 架构

```
Benchmark = Generator × Task × Agent × Evaluator × Runtime

┌──────────────────────────────────────────────────────────────┐
│  Generator                                                    │
│  dataset.parquet → yield BatchItem                            │
└───────────────────────┬──────────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────┐
│  Task: taiji_analysis                                         │
│  1. 检查实例过期 → 2. run_agent() → 3. evaluate()            │
└─────────┬────────────────────────────────┬───────────────────┘
          ▼                                ▼
┌─────────────────────┐         ┌──────────────────────────────┐
│  Agent: claude_code │         │  Evaluator: TaijiEvaluator   │
│  - 全量 38 个 skill │         │  - LLM Judge (GPT-5.5)      │
│  - 自主选择工具     │         │  - 根因是否一致 → 0/1        │
│  - CLAUDE.md 引导   │         └──────────────────────────────┘
└─────────────────────┘

Runtime: Nexus 标准镜像 + pre_start_scripts 动态安装依赖
```

### 评分规则

Binary：根因结论一致 = 1，不一致 = 0。只看最终产出，不评中间过程。

---

## 环境准备

```bash
git clone -b debug_agent_taiji_replay git@git.woa.com:nexus-agent/nexus.git
cd nexus
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev]" litellm boto3
```

---

## 准备数据集

```bash
# 从 agentd DB 提取数据集（需在有 agentd 的机器上运行）
python3 nexus/extensions/tasks/taiji_analysis/data_prep/collect_episodes.py
# 输出：episodes/dataset.json

# 如果调度节点不是数据所在机器，先拷过去
scp episodes/dataset.json <调度节点>:/path/to/nexus/episodes/

# 三种模式，默认输出 episodes/dataset.parquet（覆盖），加 --output 指定其他路径

# 全量转换（不过滤）
python3 nexus/extensions/tasks/taiji_analysis/data_prep/convert_dataset.py

# 规则过滤：只保留任务失败需要诊断的 case，排除正常完成/运维操作
python3 nexus/extensions/tasks/taiji_analysis/data_prep/convert_dataset.py --mode rule --skip-url-check
# 加 URL 连通性检查（验证实例仍可查询，慢）：
# python3 nexus/extensions/tasks/taiji_analysis/data_prep/convert_dataset.py --mode rule
# 只抽样检查部分 URL：
# python3 nexus/extensions/tasks/taiji_analysis/data_prep/convert_dataset.py --mode rule --url-check-sample 50

# 模型过滤：基于多模型结果，去掉所有模型都答对的（无区分度）
# --results-dir 为 run-batch 的模型结果保存目录（包含各模型子目录和 results.jsonl）
python3 nexus/extensions/tasks/taiji_analysis/data_prep/convert_dataset.py --mode model --results-dir /path/to/nexus_results

# 指定输出路径（不覆盖默认文件）
python3 nexus/extensions/tasks/taiji_analysis/data_prep/convert_dataset.py --mode rule --skip-url-check --output episodes/dataset_filtered.parquet
```

---

## 配置环境变量

```bash
# ── Agent LLM（被评估模型）──
export TAIJI_OPENAI_API_KEY="<APP_ID>:<APP_KEY>"
export TAIJI_OPENAI_BASE_URL="http://llm-api.model-eval.woa.com/v1"
export TAIJI_OPENAI_MODEL="openai/api_taiji_hy3"

# ── Reasoning 模式（可选）──
# 不设置则不开思考；可选值：no_think / low / high
export TAIJI_REASONING_EFFORT="high"

# ── Judge LLM（评分）──
export TAIJI_JUDGE_API_KEY="<APP_ID>:<APP_KEY>"
export TAIJI_JUDGE_BASE_URL="http://llm-api.model-eval.woa.com/v1"
export TAIJI_JUDGE_MODEL="openai/api_azure_openai_gpt-5.6-sol"
# API 风格：chat（默认，Chat Completions）或 responses（Responses API，GPT-5.6-sol 必须用）
export TAIJI_JUDGE_API_STYLE="responses"
# Reasoning effort（可选）：none/low/medium/high/xhigh/max
export TAIJI_JUDGE_REASONING_EFFORT="xhigh"

# ── 数据集 ──
export TAIJI_DATASET_PATH="./episodes/dataset.parquet"

# ── Runtime ──
export GONGFENG_RUNTIME_TOKEN="<your-gongfeng-sandbox-token>"

# ── Skill Token ──
export TAIJI_API_TOKEN="h3kcqqyYy5Uosyuyx1E5mw"
export GF_TOKEN="<your-gongfeng-pat>"
export SWANLAB_API_KEY="<your-swanlab-key>"

# ── MCP Token（注入容器，组装为 mcporter.json）──
export MCPORTER_IWIKI_TOKEN="<your-iwiki-mcp-token>"
export MCPORTER_GONGFENG_TOKEN="<your-gongfeng-mcp-token>"
export MCPORTER_TAPD_TOKEN="<your-tapd-mcp-token>"

# ── 自动安装缺失依赖（兜底） ──
export NEXUS_AUTO_INSTALL=1

# ── 可选：运行时过滤（不设则跑全量） ──
# export TAIJI_SKIP=10                    # 跳过前 N 条
# export TAIJI_LIMIT=50                   # 最多跑 N 条
# export TAIJI_INSTANCE_IDS="id1,id2"     # 只跑指定 instId（逗号分隔）
# export TAIJI_SKIP_JUDGE=1              # 跳过 Judge 评分（用于 GT 生成）
```

---

## Bedrock 透传模式（Claude Opus 思考档位）

用于评测 AWS Bedrock 上的 Claude Opus 并启用思考（thinking）能力。标准 OpenAI 协议下 `reasoning_effort` 除 `none` 外均返回 400，Bedrock 模式通过 host 侧 litellm 走 Anthropic Messages 协议实现思考档位控制。仅适用于 Claude 系模型（默认 `anthropic.claude-opus-4-8`）。

```bash
export TAIJI_BEDROCK_MODE=1
export TAIJI_OPENAI_API_KEY="<APP_ID>:<APP_KEY>"
# 注意：Bedrock 模式下不需要设置 TAIJI_OPENAI_BASE_URL 和 TAIJI_OPENAI_MODEL

# 思考档位：low / medium / high / xhigh / max（不设置或 none 则关闭思考）
# Bedrock 仅支持 thinking.type=adaptive，即模型自行决定每轮是否思考，
# effort 控制思考深度上限，不能强制每轮都产出思考内容
export TAIJI_REASONING_EFFORT="high"

# 可选覆盖（一般不需要改）：
# export TAIJI_BEDROCK_MODEL="anthropic.claude-opus-4-8"
# export TAIJI_BEDROCK_BASE_URL="http://trpc-gpt-eval.production.polaris:8080"
```

关闭：`unset TAIJI_BEDROCK_MODE` 或设为 `0`，恢复标准模式。

注意：
- 容器内 Claude Code 仍走 nexus proxy，轨迹正常捕获
- Judge LLM 不受影响，仍走 `TAIJI_JUDGE_*` 配置
- 不设 `TAIJI_BEDROCK_MODE` 则完全走原来的逻辑，向后兼容

---

## 运行

```bash
uv run nexus run-batch \
    -g nexus/extensions/tasks/taiji_analysis/generator.py \
    -d ./results/taiji_bench.db \
    --max-concurrent 100 \
    --timeout 600
```

### GT 生成模式（跳过 Judge）

只跑 Agent 产出 prediction 作为 GT 候选，不需要配置 Judge LLM：

```bash
# TAIJI_DATASET_PATH 指向新增 case 的数据集（由 add_cases/init_cases.py 产出）
# 如果要对主数据集重新生成 GT，改为 ./episodes/dataset.parquet
TAIJI_SKIP_JUDGE=1 \
TAIJI_DATASET_PATH=./new_cases/dataset.parquet \
uv run nexus run-batch \
    -g nexus/extensions/tasks/taiji_analysis/generator.py \
    -d ./gt_results/taiji_bench.db \
    --max-concurrent 10
```

跑完后 DB 中 `prediction` 即为 GT 候选（`score=-1` 表示未评判），可用 `replace_gt.py` 写入数据集。

---

## 监控进度

```bash
uv run nexus inspect-batch ./results/taiji_bench.db
```

---

## 断点续跑

```bash
# 续跑（自动跳过已成功的，不需要 -g）
uv run nexus run-batch \
    -d ./results/taiji_bench.db \
    --max-concurrent 100

# 只重跑失败的
uv run nexus run-batch \
    -d ./results/taiji_bench.db \
    --retry-only \
    --max-concurrent 100
```

---

## 替换 Ground Truth

用 run-batch 跑出的结果替换 benchmark 的 ground truth：

```bash
# 先预览（不写文件）
python3 nexus/extensions/tasks/taiji_analysis/gt_management/replace_gt.py --db ./results/taiji_bench_v3.db --dry-run

# 执行替换（更新 dataset.json 并重新生成 parquet）
python3 nexus/extensions/tasks/taiji_analysis/gt_management/replace_gt.py --db ./results/taiji_bench_v3.db
```

## 用新 GT 重新评判（无需重跑 Agent）

已有模型 prediction 结果，只想换 GT 重新 judge：

```bash
export TAIJI_JUDGE_API_KEY="$APP_ID:$APP_KEY"
export TAIJI_JUDGE_BASE_URL="http://llm-api.model-eval.woa.com/v1"
export TAIJI_JUDGE_MODEL="api_azure_openai_gpt-5.6-sol"
export TAIJI_JUDGE_API_STYLE="responses"
export TAIJI_JUDGE_REASONING_EFFORT="xhigh"

python3 nexus/extensions/tasks/taiji_analysis/gt_management/rejudge_with_new_gt.py \
    --source-results /path/to/opus4.8_xhigh_results0713 \
    --new-gt /path/to/dataset_v1_newrun_0720_390/dataset.parquet \
    --output /path/to/opus4.8_xhigh_results0720_v1 \
    --concurrency 10
```

输出目录包含 `results.jsonl` + `metrics.json` + `taiji_bench_v4.db`，可直接用于前端展示。

注意：GPT-5.6-sol 限 30 RPM，concurrency 建议 ≤ 10。

---

## 计算指标

```bash
python -m nexus.extensions.tasks.taiji_analysis.compute_metrics \
    --db ./results/taiji_bench.db \
    --output ./results/metrics.json
```

---

## 导出结果

```bash
uv run nexus export --format jsonl ./results/taiji_bench.db
uv run nexus export --format full ./results/taiji_bench.db
```

---

## 容器环境

Agent 在自定义镜像 `mirrors.tencent.com/lunahyhuang/nexus-runtime:1.4.8-debug-agent` 中运行。

镜像内预置了 `/data/workspace/.agent/skills/`（全部 40 个 skill），无需从 git 拉取。

容器启动时 `task.py` 会额外：
- 上传 `CLAUDE.md` 到容器
- 安装 `@tencent/mcporter-internal`（MCP CLI）
- 组装 `mcporter.json`（MCP 端点配置）

### 覆盖镜像内的 skills

如需临时替换 skill 而不重新打镜像，设环境变量：

```bash
export TAIJI_AGENT_SKILLS_DIR=/path/to/new/.agent
```

设置后 `task.py` 会将该目录上传到容器的 `/data/workspace/.agent`，覆盖镜像内的版本。不设则使用镜像内预置的 skills。

同理可覆盖 CLAUDE.md：

```bash
export TAIJI_CLAUDE_MD_PATH=/path/to/new/CLAUDE.md
```

---

## 环境变量说明

| 变量 | 用途 | 必填 |
|------|------|------|
| `TAIJI_OPENAI_API_KEY` | Agent LLM API Key（格式 `APP_ID:APP_KEY`） | 是 |
| `TAIJI_OPENAI_BASE_URL` | Agent LLM Endpoint | 是 |
| `TAIJI_OPENAI_MODEL` | Agent 模型 ID（需含 `openai/` 前缀） | 是 |
| `TAIJI_JUDGE_API_KEY` | Judge LLM API Key（格式 `APP_ID:APP_KEY`） | 否（默认复用 Agent） |
| `TAIJI_JUDGE_BASE_URL` | Judge LLM Endpoint | 否 |
| `TAIJI_JUDGE_MODEL` | Judge 模型 ID（需含 `openai/` 前缀） | 否 |
| `TAIJI_JUDGE_API_STYLE` | Judge API 风格：`chat` 或 `responses`（GPT-5.6-sol 用） | 否（默认 `chat`） |
| `TAIJI_JUDGE_REASONING_EFFORT` | Judge reasoning effort：`none/low/medium/high/xhigh/max` | 否 |
| `GONGFENG_RUNTIME_TOKEN` | Sandbox 容器管理 Token | 是 |
| `TAIJI_API_TOKEN` | 太极 MCP 查询 Token | 是（内置值） |
| `GF_TOKEN` | 工蜂 PAT（git clone + mcporter） | 是 |
| `SWANLAB_API_KEY` | SwanLab 图表 API Key | 否 |
| `MCPORTER_IWIKI_TOKEN` | iWiki MCP Bearer Token | 是 |
| `MCPORTER_GONGFENG_TOKEN` | 工蜂 MCP Bearer Token | 是 |
| `MCPORTER_TAPD_TOKEN` | TAPD MCP Bearer Token | 是 |
| `TAIJI_SKIP_JUDGE` | 设为 `1` 跳过 Judge 评分（用于 GT 生成跑，只产出 prediction） | 否（默认不跳过） |
| `TAIJI_BEDROCK_MODE` | 设为 `1` 启用 Bedrock 透传模式（支持思考档位） | 否（默认标准模式） |
| `TAIJI_BEDROCK_BASE_URL` | Bedrock 透传端点 | 否（默认 `http://trpc-gpt-eval.production.polaris:8080`） |
| `TAIJI_DATASET_PATH` | Parquet 数据集路径 | 否（默认 `./episodes/dataset.parquet`） |
| `TAIJI_SKIP` | 跳过前 N 条 | 否 |
| `TAIJI_LIMIT` | 最多跑 N 条 | 否 |
| `TAIJI_INSTANCE_IDS` | 逗号分隔 instId 过滤 | 否 |

---

## Token 获取

| Token | 获取方式 |
|-------|---------|
| `TAIJI_OPENAI_API_KEY` | https://model-eval.woa.com/model → 申请账号 → 获取 APP_ID:APP_KEY |
| `GONGFENG_RUNTIME_TOKEN` | 工蜂 Sandbox 资源平台申请；如已跑过 SWE-bench 可复用 |
| `GF_TOKEN` | 工蜂 → Settings → Access Tokens → 创建 PAT |
| `TAIJI_API_TOKEN` | 固定值 `h3kcqqyYy5Uosyuyx1E5mw` |
| `TAIJI_JUDGE_API_KEY` | 同 TAIJI_OPENAI_API_KEY（可复用同一个 APP_ID:APP_KEY） |
| `MCPORTER_IWIKI_TOKEN` | Knot 平台 → iWiki MCP 应用 → 创建 PAT |
| `MCPORTER_GONGFENG_TOKEN` | Knot 平台 → 工蜂 MCP 应用 → 创建 PAT |
| `MCPORTER_TAPD_TOKEN` | Knot 平台 → TAPD MCP 应用 → 创建 PAT |
| `SWANLAB_API_KEY` | SwanLab 平台 → Settings → API Keys（可选） |

---

## 预估时间

100 并发 × N 条 × 平均 3-5 分钟/条，按当前数据量约 **30-45 分钟**

---

## 常见问题

**中途失败了？** 直接用断点续跑命令，自动跳过已成功的。

**想换模型？** 改 `TAIJI_OPENAI_MODEL`，用新的 `-d` 路径重跑。

**容器配额不够？** 降低 `--max-concurrent`。
