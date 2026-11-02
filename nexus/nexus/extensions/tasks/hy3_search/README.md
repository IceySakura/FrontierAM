# HY3 Search Task

基于 Nexus 的搜索任务评估框架，用于自动化测试和评估 AI Agent 的信息检索与问答能力。

## ⚡ 快速开始

```bash
# 1. 配置环境（首次运行）
vim .env.hy3_search

# 2. 运行评估（关键参数）
python nexus/extensions/tasks/hy3_search/run_benchmarks.py \
  -b seal_0 gaia_text_only \   # 【选择 Benchmark】
  -em llm_judge \               # 【评估模式】（必选）
  -m 100 \                      # 【并发数】
  -c benchmarks.yaml             # 【配置文件】（可选）
```

## 🚀 详细说明

### 1. 环境准备
编辑项目根目录下的环境变量文件：
```bash
vim .env.hy3_search
```

### 2. 执行评估

**常用命令：**

```bash
# 【选择 Benchmark】运行指定的 benchmark（-em 为必选参数）
python nexus/extensions/tasks/hy3_search/run_benchmarks.py -b seal_0 gaia_text_only -em llm_judge

# 【控制并发】设置每个 benchmark 的并发数（默认 500）
python nexus/extensions/tasks/hy3_search/run_benchmarks.py -b seal_0 -em exact_match -m 100

# 【配置文件】使用自定义配置文件
python nexus/extensions/tasks/hy3_search/run_benchmarks.py -c benchmarks.yaml -em llm_judge

# 【模型覆盖】指定模型（最高优先级）
python nexus/extensions/tasks/hy3_search/run_benchmarks.py -b seal_0 -em llm_judge --model gpt-4o

# 【多次运行】每个 benchmark 运行 3 次，计算 avg@3
python nexus/extensions/tasks/hy3_search/run_benchmarks.py -b seal_0 -em llm_judge -k 3

# 【限制样本数】限制每个 benchmark 的数据集条目数
python nexus/extensions/tasks/hy3_search/run_benchmarks.py -b seal_0 -em llm_judge --limit 100

# 运行所有默认 benchmark
python nexus/extensions/tasks/hy3_search/run_benchmarks.py -em llm_judge

# 断点续跑（从已有数据库恢复未完成的任务）
python nexus/extensions/tasks/hy3_search/run_benchmarks.py -r -em llm_judge

# 命名实验（固定输出目录）
python nexus/extensions/tasks/hy3_search/run_benchmarks.py -n exp1 -em llm_judge

# 命名实验 + 断点续跑
python nexus/extensions/tasks/hy3_search/run_benchmarks.py -n exp1 -r -em llm_judge

# 预览模式（不实际执行）
python nexus/extensions/tasks/hy3_search/run_benchmarks.py --dry-run -em llm_judge
```

**可用的 Benchmark：**

`seal_0` `gaia_text_only` `old_gaia_text_only` `browse_comp_zh` `browse_comp_150` `browse_comp_full` `fin_search_comp_t3` `fin_search_comp_t2_and_t3` `fin_search_comp_full` `old_wide_search` `wide_search` `facts_search` `frames` `hle_150` `hy_seal_0` `hy_browse_comp_zh`

> **提示**：使用 `--dry-run` 可预览将要运行的 benchmark 列表

### 3. 查看进度

评测过程中可实时查看进度：
```bash
uv run nexus inspect-batch workspace/search_seal_0.db
```

### 4. 计算指标
导出并统计结果：
```bash
# 从数据库直接计算（推荐）
python nexus/extensions/tasks/hy3_search/compute_metrics.py --db workspace/search.db

# 或从导出的 JSONL 计算
nexus export workspace/search.db && \
python nexus/extensions/tasks/hy3_search/compute_metrics.py workspace/results.jsonl
```

---

## ⚙️ 配置说明

在 `.env.hy3_search` 中，核心配置项包括：

| 类别 | 环境变量 | 说明 |
| :--- | :--- | :--- |
| **模型** | `SEARCH_OPENAI_MODEL` | 主 Agent 模型名称 |
| | `OPENAPI_URL` | OpenAPI 请求地址 |
| | `HY_OPENAPI_KEY` | OpenAPI 密钥 |
| | `OPENAPI_ROUTE_ENV` | 路由环境（如 `response`） |
| **评测** | `JUDGE_OPENAI_BASE_URL` | LLM Judge 服务地址 |
| | `JUDGE_OPENAI_API_KEY` | LLM Judge API 密钥 |
| | `JUDGE_OPENAI_MODEL` | LLM Judge 模型 |
| | `JUDGE_OPENAI_CLIENTS_NAME` | LLM Judge 客户端名称 |
| | `JUDGE_USE_TOOL_CALLING` | Judge 是否使用 tool calling |
| **推理** | `SEARCH_TEMPERATURE` | 采样温度 |
| | `SEARCH_REASONING_EFFORT` | 推理力度（如 `high`） |
| | `SEARCH_INTERLEAVED_THINKING` | 是否启用交错思考 |
| | `SEARCH_MAX_TOKENS` | 最大 token 数 |
| | `SEARCH_ENABLE_SUMMARY` | 是否启用摘要 |
| **数据** | `SEARCH_BENCHMARK_DATA_DIR` | Parquet 数据集目录 |
| **控制** | `SEARCH_AGENT_NAME` | Agent 名称 |
| | `SEARCH_AGENT_MAX_RETRIES` | Agent 最大重试次数 |
| | `SEARCH_TIMEOUT_PER_RUN` | 单次运行超时（秒） |
| | `SEARCH_SKIP` | 跳过前 N 条数据 |
| | `SEARCH_LIMIT` | 限制评估样本数量 |
| | `SEARCH_RUN_ORACLE` | 是否运行 Oracle 模式 |
| | `DISABLE_CONTENT_FILTER` | 是否禁用内容过滤 |

---

## 📊 评估机制

### 评估模式
通过命令行参数 `-em/--eval-mode`（必选）指定评估模式：
- `exact_match`: 精确匹配标准答案。
- `llm_judge`: 使用 LLM 进行语义等价判断（推荐）。
- `reward_service`: 使用奖励服务进行评估。
- `wide_search`: 广域搜索评估（使用 `wide_search_evaluation` 子模块）。

### 指标输出
执行 `compute_metrics.py` 后，你将获得：
- **Accuracy**: 整体正确率。
- **By Source**: 按数据来源分类的统计指标。
- **Avg Time**: 平均每个样本的响应时长。
- **avg@k / pass@k**: 多次运行时的聚合指标（通过 `-k` 参数启用）。

---

## 📂 目录结构
```
hy3_search/
├── __init__.py               # 模块入口，导出 hy3_search_task
├── task.py                   # 任务定义与执行逻辑
├── generator.py              # 批量任务生成器
├── evaluator.py              # 评估策略（EM、LLM Judge、Reward Service、Wide Search）
├── prompt.py                 # 评估 Prompt 模板（Judge & Reward Service）
├── compute_metrics.py        # 指标汇总统计
├── reporting.py              # 结果报告与摘要输出（Rich 表格 / Excel / JSON）
├── utils.py                  # 通用工具函数
├── run_benchmarks.py         # 多 Benchmark 并发执行脚本（推荐入口）
├── benchmarks.yaml.example   # 配置文件示例
└── wide_search_evaluation/   # 广域搜索评估子模块
    ├── __init__.py
    ├── data_loader.py        # 数据加载
    ├── evaluation.py         # 评估逻辑
    └── metric_utils.py       # 指标计算工具
```

---

## 🔧 run_benchmarks.py 详细说明

### 命令行参数

**核心参数：**

| 参数 | 简写 | 说明 | 默认值 |
| :--- | :--- | :--- | :--- |
| `--benchmarks` | `-b` | **选择 Benchmark**（如 `seal_0 gaia_text_only`） | 全部 |
| `--eval-mode` | `-em` | **评估模式**（`exact_match` / `llm_judge` / `reward_service` / `wide_search`） | **必选** |
| `--max-concurrent` | `-m` | **控制并发数**（每个 benchmark） | 500 |
| `--config` | `-c` | **自定义配置文件**（YAML 路径） | 内置默认 |
| `--model` | | **覆盖模型**（最高优先级） | 环境变量 |
| `--runs` | `-k` | **运行次数**（启用 avg@k / pass@k） | 1 |

**其他参数：**

| 参数 | 简写 | 说明 | 默认值 |
| :--- | :--- | :--- | :--- |
| `--name` | `-n` | 实验名称（固定目录：`benchmark-run-{name}`） | 无 |
| `--output-dir` | `-o` | 输出目录（覆盖 `--name`） | 自动生成 |
| `--timeout` | `-t` | 单任务超时（秒） | 2400 |
| `--benchmark-parallel` | `-p` | 同时运行的 benchmark 数量 | 1 |
| `--resume` | `-r` | 从已有数据库恢复未完成的任务 | False |
| `--env-file` | `-e` | 环境变量文件路径 | `.env.hy3_search` |
| `--verbose` | `-v` | 详细日志 | False |
| `--dry-run` | | 预览模式，不执行 | False |
| `--no-table` | | 禁用 Rich 表格输出，使用纯文本 | False |
| `--agent-max-retries` | | Agent 最大重试次数 | 环境变量 |
| `--instance-ids` | | 指定数据集 instance_id（逗号/空格分隔） | 无 |
| `--run-ids` | | 指定 batch run_id（逗号/空格分隔） | 无 |
| `--limit` | | 限制每个 benchmark 的数据集条目数 | 无 |

### 配置文件格式（可选）

创建 `benchmarks.yaml` 自定义数据集路径：

```yaml
benchmarks:
  seal_0: /path/to/sealqa_seal_0.parquet
  gaia_text_only: /path/to/gaia_text_only.parquet
  custom_benchmark: /path/to/your_data.parquet
```

然后使用：`python run_benchmarks.py -c benchmarks.yaml -em llm_judge`

### 输出结构

```
workspace/benchmark-run-{model}/
├── search_seal_0.db              # SQLite 数据库
├── metrics_seal_0.json           # 评估指标
├── traj_seal_0.jsonl             # 轨迹文件（每行一个任务）
├── search_gaia_text_only.db
├── metrics_gaia_text_only.json
├── traj_gaia_text_only.jsonl
└── RUN_SUMMARY.json              # 运行汇总
```

### 环境变量自动加载

脚本会自动加载 `.env.hy3_search` 中的环境变量（除了 `SEARCH_DATASET_PATH`），无需手动 `source`。使用内置解析器，无需额外依赖。
