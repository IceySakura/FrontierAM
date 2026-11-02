# MCP Benchmark Universe（Gongfeng）

相关文档：

- [总览](../README.md)
- [Atlas 详情](atlas.md)
- [Toolathlon 详情](toolathlon.md)

## 0. 数据准备

Universe 现在先走 prepare，再走筛题 / generator：

```bash
uv run python -m nexus.extensions.tasks.mcp_benchmark.prepare_data --datasets universe
```

如果本地 `workspace/mcp/datasets/universe` 已经准备好，只想重建 manifest：

```bash
uv run python -m nexus.extensions.tasks.mcp_benchmark.prepare_data --datasets universe --skip-sync
```

- 上游题目来源：`workspace/mcp/datasets/universe/mcpuniverse/benchmark/configs/mcpuniverse/**/*.json`
- prepare 产物：`workspace/mcp/converted/universe.jsonl`
- 后续 `universe_selection.py`、`datasets.py`、`generator.py` 都只读取这个 JSONL manifest，不再直接扫描原始题目目录生成 case 列表

## 1. 两种运行模式

- `api-free`：面向“没有 original 额外凭据，或者先想低成本跑一批题”时的 best effort 路径。它会优先走当前 benchmark 已经接好的 mock / shim，但不是“完全离线”模式。
- `original`：按 Universe 原始 MCP server / API 链路运行。更推荐按你手里已有的 API 族缩子集运行。

当前 Universe 快照来自：

```bash
uv run python -m nexus.extensions.tasks.mcp_benchmark.universe_selection summary
```

当前结果：

- Universe 总题量：`232`
- `api-free` 可跑：`199`
- `api-free` 当前的主要缺口：`github` 相关题目有 `33` 道被排除，因为当前 mock 路径仍会 fallback 到 live GitHub

为什么 `api-free` 不是 `232`：

- 当前 benchmark 已经把 `google-search`、`google-maps`、`notion` 这些 Universe 高占比外部服务接到本地 shim 上。
- 但 `github` 在 `api-free` 下仍会走 live GitHub，因此在“没有 GitHub token”的前提下必须排除。

## 2. 安装

```bash
uv pip install -e ".[dev]"
nexus modules install mcp_benchmark
uv pip install pyarrow boto3
```

- `pyarrow` 是当前 benchmark 公共依赖的一部分。
- `boto3` 是当前 LiteLLM 路径在这个环境里的实际依赖。

## 3. Gongfeng 基线配置

下面这组值来自已经跑通过的 Gongfeng batch 配置，可以直接作为默认基线：

```bash
export MCP_BENCHMARK_DATASETS=universe

export MCP_BENCHMARK_RUNTIME_PROVIDER_NAME=gongfeng
export MCP_BENCHMARK_RUNTIME_BASE_URL=http://hyrl-sandbox.prod.woa.com:8052
export MCP_BENCHMARK_RUNTIME_IMAGE=current
export MCP_BENCHMARK_MAIN_IMAGE=mirrors.tencent.com/hunyuan_yanguan/mcp:main-feat-mcp-ae1206e
export MCP_BENCHMARK_RUNTIME_REQUEST_TIMEOUT=3600
export MCP_BENCHMARK_RUNTIME_PROVIDER_CLUSTER=gongfeng
export MCP_BENCHMARK_RUNTIME_TOKEN=<your_gongfeng_token>
export MCP_BENCHMARK_RUNTIME_CPU=2
export MCP_BENCHMARK_RUNTIME_MEMORY=10

export MCP_BENCHMARK_OPENAI_CLIENTS_NAME=litellm
export MCP_BENCHMARK_OPENAI_BASE_URL=http://trpc-gpt-eval.production.polaris:8080
export MCP_BENCHMARK_OPENAI_API_KEY=<your_llm_api_key>
export MCP_BENCHMARK_OPENAI_MODEL=anthropic.claude-sonnet-4-6
```

`MCP_BENCHMARK_RUNTIME_TOKEN` 是 Gongfeng 必填项。

## 4. Universe 需要哪些额外凭据 / 配置

下面这张表按 Universe 每道题的 `key_required_servers` 统计；一道题命中多个服务时会重复计数。

如需重算：

```bash
uv run python -m nexus.extensions.tasks.mcp_benchmark.universe_selection api-counts --format table
```

上游原始说明优先参考：

- [Universe `README.md`](../../../../../workspace/mcp/datasets/universe/README.md)
- [Universe `.env.example`](../../../../../workspace/mcp/datasets/universe/.env.example)

| 服务来源 | Universe 题量 | 需要准备的字段 / 工件 | 获取方式 / 备注 |
| --- | --- | --- | --- |
| Google Search | `55 (23.7%)` | `serper_api_key` | Universe 上游 README 同时提到 `SERP_API_KEY` / `SERPER_API_KEY`。在当前 benchmark 里，准备一份 Serper / 搜索 key 并注入 `serper_api_key` 即可。 |
| Google Maps | `45 (19.4%)` | `google_cloud_console_api_key` | 按 Universe 上游 README 的 Google Maps 条目在 Google Cloud Console 创建 key。 |
| GitHub | `33 (14.2%)` | `github_token` | 按 Universe 上游 README 的 GitHub 条目准备 PAT。若自动推导用户名失败，再额外补 `github_personal_account_name` / `github_account_name`。 |
| Notion | `10 (4.3%)` | `notion_integration_key`, `notion_integration_key_eval`, `source_notion_page_url`, `eval_notion_page_url` | Universe 上游 README 要求 `NOTION_API_KEY` + `NOTION_ROOT_PAGE`。当前 benchmark 更适合直接提供 source/eval 两套 token 与页面 URL，由其自动导出 root page。 |

## 5. 查题与筛题

Universe 的本地查询入口：

```bash
uv run python -m nexus.extensions.tasks.mcp_benchmark.universe_selection summary
uv run python -m nexus.extensions.tasks.mcp_benchmark.universe_selection api-counts --format table
uv run python -m nexus.extensions.tasks.mcp_benchmark.universe_selection query --mode api-free --format table
uv run python -m nexus.extensions.tasks.mcp_benchmark.universe_selection query --contains-servers github --format table
```

generator 内可用的 Universe 筛题参数：

| 变量名 | 含义 |
| --- | --- |
| `MCP_BENCHMARK_UNIVERSE_MODE` | Universe 用户模式，只能是 `api-free` 或 `original` |
| `MCP_BENCHMARK_UNIVERSE_CONTAINS_SERVERS` | 只保留包含指定 server 的题 |
| `MCP_BENCHMARK_UNIVERSE_CONTAINS_MATCH` | `all` 或 `any` |
| `MCP_BENCHMARK_UNIVERSE_ALLOWED_SERVERS` | 只保留其完整 server 集合是给定集合子集的题 |
| `MCP_BENCHMARK_UNIVERSE_ALLOWED_KEYED_SERVERS` | 只保留其“额外凭据依赖集合”是给定集合子集的题 |
| `MCP_BENCHMARK_UNIVERSE_ALLOWED_API_SERVERS` | `MCP_BENCHMARK_UNIVERSE_ALLOWED_KEYED_SERVERS` 的别名 |
| `MCP_BENCHMARK_CASE_IDS` | 在 Universe 选择器之后再做精确 case_id 钉死 |
| `MCP_BENCHMARK_SKIP` / `MCP_BENCHMARK_LIMIT` | 分页和 smoke 用 |

几个常用例子：

```bash
# 看 api-free 现在能跑哪些题
uv run python -m nexus.extensions.tasks.mcp_benchmark.universe_selection query \
  --mode api-free \
  --format table

# 只看包含 github 的题
uv run python -m nexus.extensions.tasks.mcp_benchmark.universe_selection query \
  --contains-servers github \
  --contains-match any \
  --format table

# 只保留“额外凭据依赖集合”属于 {github,google-search} 的题
uv run python -m nexus.extensions.tasks.mcp_benchmark.universe_selection query \
  --allowed-keyed-servers github,google-search \
  --format table

# 直接拿 case_id，后面喂给 MCP_BENCHMARK_CASE_IDS
uv run python -m nexus.extensions.tasks.mcp_benchmark.universe_selection query \
  --contains-servers github \
  --allowed-keyed-servers github \
  --format case-ids
```

怎么理解这几个筛题参数：

- 你“只有少量 token”时，优先用 `MCP_BENCHMARK_UNIVERSE_ALLOWED_KEYED_SERVERS` 控制远端 API 族。
- 你想把题目依赖的 MCP server 也一起收紧时，再用 `MCP_BENCHMARK_UNIVERSE_ALLOWED_SERVERS`。
- 你只想挑某类 server 题，例如 GitHub 题，再叠加 `MCP_BENCHMARK_UNIVERSE_CONTAINS_SERVERS=github`。

## 6. `api-free` 全流程

`api-free` 的定位是：在缺少 original 额外凭据时，先把当前 benchmark 已经能 best-effort 跑通的 Universe 子集跑起来。它不是“严格纯 mock”。

### 6.1 跑当前全部 `api-free` 可跑集

```bash
export MCP_BENCHMARK_DATASETS=universe
export MCP_BENCHMARK_UNIVERSE_MODE=api-free

unset MCP_BENCHMARK_UNIVERSE_CONTAINS_SERVERS
unset MCP_BENCHMARK_UNIVERSE_CONTAINS_MATCH
unset MCP_BENCHMARK_UNIVERSE_ALLOWED_SERVERS
unset MCP_BENCHMARK_UNIVERSE_ALLOWED_KEYED_SERVERS
unset MCP_BENCHMARK_UNIVERSE_ALLOWED_API_SERVERS
unset MCP_BENCHMARK_CASE_IDS
unset MCP_BENCHMARK_SKIP
unset MCP_BENCHMARK_LIMIT

export MCP_BENCHMARK_GENERATION_REPORT=workspace/mcp/tmp/generation_universe_apifree.json

uv run nexus run-batch \
  -g nexus/extensions/tasks/mcp_benchmark/generator.py \
  -d workspace/mcp/tmp/batch_universe_apifree.db \
  --max-concurrent 20 \
  --export-format full
```

### 6.2 跑一个 `api-free` 子集

```bash
export MCP_BENCHMARK_DATASETS=universe
export MCP_BENCHMARK_UNIVERSE_MODE=api-free
export MCP_BENCHMARK_UNIVERSE_ALLOWED_SERVERS=google-maps
export MCP_BENCHMARK_LIMIT=20

uv run nexus run-batch \
  -g nexus/extensions/tasks/mcp_benchmark/generator.py \
  -d workspace/mcp/tmp/batch_universe_apifree_subset.db \
  --max-concurrent 20 \
  --export-format full
```

### 6.3 已验证的 `api-free` smoke

```bash
export MCP_BENCHMARK_DATASETS=universe
export MCP_BENCHMARK_UNIVERSE_MODE=api-free
export MCP_BENCHMARK_UNIVERSE_ALLOWED_SERVERS=google-maps
export MCP_BENCHMARK_CASE_IDS=google_maps_task_0031
export MCP_BENCHMARK_LIMIT=1
export MCP_BENCHMARK_GENERATION_REPORT=workspace/mcp/tmp/generation_universe_apifree_maps_verify.json

uv run nexus run-batch \
  -g nexus/extensions/tasks/mcp_benchmark/generator.py \
  -d workspace/mcp/tmp/batch_universe_apifree_maps_verify.db \
  --max-concurrent 1 \
  --export-format full
```

验证结果：

- `db`: `workspace/mcp/tmp/batch_universe_apifree_maps_verify.db`
- `run_id`: `universe-mock-google_maps_task_0031`
- `status`: `success`
- `passed`: `0`
- `score`: `0.0`

注：这里的 `run_id` 会沿用内部 `universe-mock-*` server_mode 命名；对外用户模式仍统一称 `api-free`。
注：这里的 smoke 主要用于验证 generator / runtime / MCP / evaluator 链路可执行；`passed` / `score` 仍取决于当前 agent 在该样例上的表现。

## 7. `original` 全流程

Universe 的 `original` 更适合“按你已有的 API 族缩成子集后再跑”。

### 7.1 只跑 GitHub 子集

```bash
export MCP_BENCHMARK_DATASETS=universe
export MCP_BENCHMARK_UNIVERSE_MODE=original
export MCP_BENCHMARK_UNIVERSE_CONTAINS_SERVERS=github
export MCP_BENCHMARK_UNIVERSE_CONTAINS_MATCH=any
export MCP_BENCHMARK_UNIVERSE_ALLOWED_KEYED_SERVERS=github
export MCP_BENCHMARK_LIMIT=20

uv run nexus run-batch \
  -g nexus/extensions/tasks/mcp_benchmark/generator.py \
  -d workspace/mcp/tmp/batch_universe_original_github_subset.db \
  --max-concurrent 20 \
  --export-format full
```

### 7.2 只跑 Search + Maps 子集

```bash
export MCP_BENCHMARK_DATASETS=universe
export MCP_BENCHMARK_UNIVERSE_MODE=original
export MCP_BENCHMARK_UNIVERSE_ALLOWED_KEYED_SERVERS=google-search,google-maps

uv run nexus run-batch \
  -g nexus/extensions/tasks/mcp_benchmark/generator.py \
  -d workspace/mcp/tmp/batch_universe_original_search_maps.db \
  --max-concurrent 20 \
  --export-format full
```

### 7.3 用 case_id 精确钉死

```bash
export MCP_BENCHMARK_CASE_IDS="$(
  uv run python -m nexus.extensions.tasks.mcp_benchmark.universe_selection query \
    --contains-servers github \
    --allowed-keyed-servers github \
    --format case-ids
)"
```

### 7.4 已验证的 `original` smoke

```bash
export MCP_BENCHMARK_DATASETS=universe
export MCP_BENCHMARK_UNIVERSE_MODE=original
export MCP_BENCHMARK_UNIVERSE_CONTAINS_SERVERS=github
export MCP_BENCHMARK_UNIVERSE_CONTAINS_MATCH=any
export MCP_BENCHMARK_UNIVERSE_ALLOWED_KEYED_SERVERS=github
export MCP_BENCHMARK_CASE_IDS=github_task_0007
export MCP_BENCHMARK_LIMIT=1
export MCP_BENCHMARK_GENERATION_REPORT=workspace/mcp/tmp/generation_universe_original_verify.json

uv run nexus run-batch \
  -g nexus/extensions/tasks/mcp_benchmark/generator.py \
  -d workspace/mcp/tmp/batch_universe_original_verify.db \
  --max-concurrent 1 \
  --export-format full
```

验证结果：

- `db`: `workspace/mcp/tmp/batch_universe_original_verify.db`
- `run_id`: `universe-api-github_task_0007`
- `status`: `success`
- `passed`: `0`
- `score`: `0.5`

注：这里的 `run_id` 会沿用内部 `universe-api-*` server_mode 命名；对外用户模式仍统一称 `original`。
注：这里的 smoke 主要用于验证 generator / runtime / MCP / evaluator 链路可执行；`passed` / `score` 仍取决于当前 agent 在该样例上的表现。

## 8. 每题依赖矩阵

每一道题的依赖矩阵见：

- [universe_api_matrix.md](universe_api_matrix.md)

重生成命令：

```bash
uv run python -m nexus.extensions.tasks.mcp_benchmark.universe_selection matrix \
  --output nexus/extensions/tasks/mcp_benchmark/docs/universe_api_matrix.md
```

矩阵里最值得看的几列：

- `依赖 MCP 服务`
- `需要额外凭据的服务`
- `凭据字段名`
- `api-free 可跑`

如果你只想看 API / 凭据来源统计而不是逐题矩阵：

```bash
uv run python -m nexus.extensions.tasks.mcp_benchmark.universe_selection api-counts --format table
```
