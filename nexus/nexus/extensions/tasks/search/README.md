# Search Task

基于 Nexus 的搜索任务评估框架，用于自动化测试和评估 AI Agent 的信息检索与问答能力。

## 快速开始

```bash
# 1. 复制并编辑配置文件
cp nexus/extensions/tasks/search/search_config.yaml.example search_config.yaml
vim search_config.yaml  # 填入 API Key、模型名等

# 2. 运行评估
python nexus/extensions/tasks/search/run_benchmarks.py -c search_config.yaml

# 3. 只跑部分 benchmark（调试用）
python nexus/extensions/tasks/search/run_benchmarks.py -c search_config.yaml -b seal_0 --limit 5
```

一个 YAML 文件包含所有配置（LLM 客户端、工具凭证、Agent 参数、Prompt、评估模式、Benchmark 列表），无需 `source .env.search`。

> **数据集路径**：配置文件中的 benchmark 路径默认指向共享存储。 如需在其他机器运行，设置环境变量即可切换全部路径：
> ```bash
> export SEARCH_BENCHMARK_DATA_DIR=/your/local/data
> ```

## 配置文件

### 结构总览

`search_config.yaml` 包含 7 个 section，均为可选：

```yaml
clients:        # LLM 客户端（main + judge）
evaluation:     # 评估模式（llm_judge / exact_match / reward_service）
prompt:         # System prompt（支持 {today_date} 和 {finish_instruction} 占位符）
toolset:        # 工具集定义（v1 / v2 / 自定义）
agent:          # Agent 参数（max_iterations、is_training 等）
tool_configs:   # 工具凭证（serper、jina、jupyter）
benchmarks:     # Benchmark 数据集列表
```

### 环境变量插值

所有字符串值支持 `${VAR}` 和 `${VAR:-default}` 语法，密钥可以留在环境变量中：

```yaml
tool_configs:
  google_serper_search:
    serper_app_key: ${SEARCH_SERPER_APP_KEY}
    serper_url: ${SEARCH_SERPER_URL:-http://default-url:8080/search}
```

### 配置优先级

- 提供 YAML 时：YAML 是唯一配置源，环境变量构建器被完全跳过，不混合
- 不提供 YAML 时：走原有的 `.env.search` 环境变量逻辑，完全向后兼容

### 完整示例

参见 [`search_config.yaml.example`](search_config.yaml.example)。

## 运行评估

### 常用命令

```bash
# 跑全部 benchmark
python nexus/extensions/tasks/search/run_benchmarks.py -c search_config.yaml

# 只跑指定的 benchmark
python nexus/extensions/tasks/search/run_benchmarks.py -c search_config.yaml -b seal_0 gaia_text_only

# 限制样本数（快速调试）
python nexus/extensions/tasks/search/run_benchmarks.py -c search_config.yaml -b seal_0 --limit 5

# 预览模式（不实际执行）
python nexus/extensions/tasks/search/run_benchmarks.py -c search_config.yaml --dry-run

# 断点续跑（命名实验）
python nexus/extensions/tasks/search/run_benchmarks.py -c search_config.yaml -r -n my_experiment

# avg@3（每个 benchmark 跑 3 次取平均）
python nexus/extensions/tasks/search/run_benchmarks.py -c search_config.yaml -k 3
```

### 不使用 YAML（传统方式）

仍然支持通过环境变量配置，与改动前完全一致：

```bash
source .env.search
python nexus/extensions/tasks/search/run_benchmarks.py -b seal_0
```

### 可用 Benchmark

使用 `-b` 选择子集（如 `-b seal_0 gaia_text_only`），不指定 `-b` 时跑配置中全部 benchmark。

| 名称 | 数据集 | 说明 |
| :--- | :--- | :--- |
| `seal_0` | sealqa_seal_0.parquet | SealQA 核心集 |
| `seal_0_260320` | sealqa_seal_0_260320.parquet | SealQA 更新版 |
| `gaia_text_only` | gaia_text_only.parquet | GAIA 纯文本 |
| `browse_comp_150` | browse_comp_150.parquet | BrowseComp 150 题子集 |
| `browse_comp_zh` | browse_comp_zh.parquet | BrowseComp 中文 |
| `browse_comp_full` | browse_comp_full.parquet | BrowseComp 全集 |
| `fin_search_comp_t3` | fin_search_comp_t3.parquet | FinSearchComp T3 |
| `fin_search_comp_t2_and_t3` | fin_search_comp_t2_and_t3.parquet | FinSearchComp T2+T3 |
| `fin_search_comp_full` | fin_search_comp_full.parquet | FinSearchComp 全集 |
| `wide_search` | wide_search.parquet | 结构化表格搜索 |
| `facts_search` | facts_search.parquet | 事实检索 |
| `frames` | frames.parquet | FRAMES |
| `hle_150` | hle_150.parquet | HLE 150 题子集 |

### 命令行参数

| 参数 | 简写 | 说明 | 默认值 |
| :--- | :--- | :--- | :--- |
| `--config` | `-c` | YAML 配置文件路径 | 无（走 env var） |
| `--benchmarks` | `-b` | 选择 benchmark 子集 | 配置文件中全部 |
| `--max-concurrent` | `-m` | 每个 benchmark 的并发数 | 500 |
| `--limit` | | 每个 benchmark 限制样本数 | 无限制 |
| `--name` | `-n` | 实验名（固定输出目录） | 自动生成 |
| `--resume` | `-r` | 断点续跑 | False |
| `--runs` | `-k` | 每个 benchmark 运行次数（avg@k） | 1 |
| `--model` | | 覆盖主模型名 | YAML 中配置 |
| `--benchmark-parallel` | `-p` | 同时运行的 benchmark 数量 | 1 |
| `--timeout` | `-t` | 每个任务的超时（秒） | 7200 |
| `--dry-run` | | 预览模式 | False |
| `--verbose` | `-v` | 详细日志 | False |

## 查看进度

```bash
uv run nexus inspect-batch workspace/benchmark-run-{name}/search_{benchmark}.db
```

## 计算指标

```bash
# 从数据库直接计算（推荐）
python nexus/extensions/tasks/search/compute_metrics.py --db workspace/benchmark-run-{name}/search_{benchmark}.db

# 从导出的 JSONL 计算
uv run nexus export workspace/search.db
python nexus/extensions/tasks/search/compute_metrics.py workspace/results.jsonl
```

## 评估模式

在 YAML 的 `evaluation.mode` 中指定：

| 模式 | 说明 |
| :--- | :--- |
| `llm_judge` | LLM 语义等价判断（推荐，需要 judge 客户端） |
| `exact_match` | 精确字符串匹配 |
| `reward_service` | 外部打分服务 |
| `wide_search` | 结构化表格评估（从数据集自动检测） |

## 输出结构

```
workspace/benchmark-run-{name}/
├── search_{benchmark}.db        # SQLite 数据库（完整结果）
├── metrics_{benchmark}.json     # 评估指标
├── traj_{benchmark}.jsonl       # 轨迹文件（每行一个任务）
├── trace_{benchmark}.jsonl      # 详细执行追踪
├── RUN_SUMMARY.json             # 运行汇总
└── RUN_SUMMARY.html             # HTML 格式汇总
```

## 源码结构

| 文件 | 说明 |
| :--- | :--- |
| `config.py` | YAML 配置加载与解析 |
| `generator.py` | 批量任务生成器（从 YAML 或 env var 构建 RunInput） |
| `task.py` | 任务定义与执行逻辑 |
| `evaluator.py` | 评估策略（exact_match / llm_judge / reward_service / wide_search） |
| `compute_metrics.py` | 指标汇总统计 |
| `run_benchmarks.py` | 多 Benchmark 并发执行脚本 |
| `reporting.py` | 结果报告生成（JSON / HTML） |
| `reevaluate_traj.py` | 对已有轨迹重新评估 |
| `search_config.yaml.example` | 配置文件模板 |
