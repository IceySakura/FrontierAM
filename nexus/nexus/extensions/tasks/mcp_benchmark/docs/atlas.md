# MCP Benchmark Atlas（Gongfeng）

相关文档：

- [总览](../README.md)
- [Toolathlon 详情](toolathlon.md)
- [Universe 详情](universe.md)

## 0. 数据准备

Atlas 现在先走 prepare，再走筛题 / generator：

```bash
uv run python -m nexus.extensions.tasks.mcp_benchmark.prepare_data --datasets atlas
```

如果本地 `workspace/mcp/datasets/atlas` 已经准备好，只想重建 manifest：

```bash
uv run python -m nexus.extensions.tasks.mcp_benchmark.prepare_data --datasets atlas --skip-sync
```

- 上游题目来源：`ScaleAI/MCP-Atlas` 的 `MCP-Atlas.parquet`
- parquet 本地缓存：`workspace/mcp/datasets/atlas/hf_cache/MCP-Atlas.parquet`
- prepare 产物：`workspace/mcp/converted/atlas.jsonl`
- 后续 `atlas_selection.py`、`datasets.py`、`generator.py` 都只读取这个 JSONL manifest，不再直接读取 parquet 生成 case 列表

## 1. 两种运行模式

- `api-free`：面向“没有 original 私有凭据，或者手里只有很少 token”时的 best effort 路径。它会优先走当前 benchmark 已经接好的 mock / shim；但题目仍然可能访问无需私有 token 的公共 API、公共网页或公共数据源，所以它不是“完全离线”或“完全不用 API”的模式。
- `original`：按 Atlas 原始 MCP server / API 链路运行。题目依赖的私有凭据没准备好，就会在 generator 里被自动过滤掉。

当前 Atlas 快照来自：

```bash
uv run python -m nexus.extensions.tasks.mcp_benchmark.atlas_selection summary
```

当前结果：

- Atlas 总题量：`500`
- `api-free` 可跑：`341`
- `api-free` 当前的主要缺口：`github` 相关题目仍有 `159` 道被排除，因为 Atlas 的 GitHub mock 目前还会 fallback 到 live GitHub，并继续要求 `github_token`

为什么 `api-free` 不等于“完全无私有凭据题目”：

- Atlas 原始数据集只把“完全不用 key 的默认 server”当作无凭据模式。
- 当前这个 benchmark 在 Atlas mock 路径上额外补了多条 mock / shim 链路，所以即便题目原始依赖了 Airtable、Notion、Slack、MongoDB 等私有 API，只要这条链路已经被本地替代，仍然可以进入 `api-free`。

## 2. 安装

```bash
uv pip install -e ".[dev]"
nexus modules install mcp_benchmark
uv pip install pyarrow boto3
```

- `pyarrow` 用来读取 Atlas 数据集。
- `boto3` 是当前 LiteLLM 路径在这个环境里的实际依赖。

## 3. Gongfeng 基线配置

下面这组值来自已经跑通过的 Atlas Gongfeng batch 配置，可以直接作为默认基线：

```bash
export MCP_BENCHMARK_DATASETS=atlas

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

Atlas `api-free` 在 Gongfeng 上如果命中 `mongodb` 题目，会自动为 `atlas-mongodb` service 选择一个已验证可用的 MongoDB mock 镜像；一般不需要额外设置。只有当你想手动覆盖时，才需要显式设置 `MCP_BENCHMARK_ATLAS_MONGODB_IMAGE`。

## 4. Atlas 需要哪些私有 API

下面这张表按 Atlas 每道题的 `key_required_servers` 统计；一道题命中多个私有 API 时会重复计数。

如需重算：

```bash
uv run python -m nexus.extensions.tasks.mcp_benchmark.atlas_selection api-counts --format table
```

上游原始说明优先参考：

- [Atlas `env.template`](../../../../../workspace/mcp/datasets/atlas/env.template)
- [Atlas `data_exports/README.md`](../../../../../workspace/mcp/datasets/atlas/data_exports/README.md)

| API 来源 | Atlas 题量 | 需要填写的字段名 | 是否还要准备样例数据 | 获取方式 / 备注 |
| --- | --- | --- | --- | --- |
| GitHub | `159 (31.8%)` | `github_token` | 否 | 按 Atlas 原始 `env.template` 的 GitHub 条目准备 PAT。少 token 时，如果你只有 GitHub，就用 `original` 加 `MCP_BENCHMARK_ATLAS_ALLOWED_KEYED_SERVERS=github` 跑 GitHub 子集。 |
| Oxylabs | `94 (18.8%)` | `oxylabs_username`, `oxylabs_password` | 否 | 按 Atlas 原始 `env.template` 的 Oxylabs 条目开通 Oxylabs 并从 dashboard 取用户名 / 密码。 |
| Airtable | `92 (18.4%)` | `airtable_api_key` | 是 | 先按 [Atlas `data_exports/README.md`](../../../../../workspace/mcp/datasets/atlas/data_exports/README.md) 里的 Airtable 步骤复制 shared base，再按 `env.template` 对应说明申请 API key。 |
| Weather Data | `86 (17.2%)` | `weather_api_key` | 否 | Atlas 原始 `env.template` 这条链路指向 `weatherapi.com`。没有天气数据 key 时，先排除 `weather-data` 子集。 |
| MongoDB | `79 (15.8%)` | `mongodb_connection_string` | 是 | 先创建 MongoDB Atlas 集群并拿 connection URI，再把 `workspace/mcp/datasets/atlas/data_exports/mongo_dump_video_game_store-UNZIP-FIRST.zip` 解压后用 `mongorestore` 导入。 |
| Exa | `77 (15.4%)` | `exa_api_key` | 否 | 按 Atlas 原始 `env.template` 的 Exa 条目开通 Exa 并创建 API key。 |
| National Parks | `76 (15.2%)` | `nps_api_key` | 否 | 按 Atlas 原始 `env.template` 的 NPS developer 条目申请 API key。 |
| Alchemy | `69 (13.8%)` | `alchemy_api_key` | 否 | 按 Atlas 原始 `env.template` 的 Alchemy 条目创建 app 并复制 API key。 |
| Notion | `67 (13.4%)` | `notion_integration_key_eval` 或 `notion_integration_key` | 是 | 先导入 `workspace/mcp/datasets/atlas/data_exports/mcp-atlas-notion-data.zip`，再创建 Notion Internal integration，拿到 integration secret。实操时最好确保 integration 已被授权访问导入后的页面 / 数据库。 |
| Brave Search | `63 (12.6%)` | `brave_api_key` | 否 | 按 Atlas 原始 `env.template` 的 Brave Search API 条目申请 key。 |
| E2B | `61 (12.2%)` | `e2b_api_key` | 否 | 按 Atlas 原始 `env.template` 的 E2B 条目创建 API key。 |
| Slack | `60 (12.0%)` | `slack_mcp_xoxc_token`, `slack_mcp_xoxd_token` | 是 | 先在你自己的 Slack workspace 导入 `workspace/mcp/datasets/atlas/data_exports/slack_mcp_eval_export.zip`。当前 Atlas 集成要求的是 `xoxc/xoxd` 会话凭据，不是普通 bot token；获取方式以上游 `slack-mcp-server` 的认证文档为准。 |
| Twelve Data | `60 (12.0%)` | `twelve_data_api_key` | 否 | 按 Atlas 原始 `env.template` 的 Twelve Data 条目申请 key。 |
| Lara Translate | `53 (10.6%)` | `lara_access_key_id`, `lara_access_key_secret` | 否 | 按 Atlas 原始 `env.template` 的 LARA Translate 条目申请 access key / secret。 |
| Google Workspace | `50 (10.0%)` | `google_client_id`, `google_client_secret`, `google_refresh_token` | 是 | 先把 `workspace/mcp/datasets/atlas/data_exports/calendar_mcp_eval_export.zip` 解压为 `.ics` 并导入 Google Calendar，再按 `@geobio/google-workspace-server` 的前置步骤在 Google Cloud Console 开 Gmail / Calendar API、创建 OAuth client，并换出 refresh token。 |
| Google Maps | `38 (7.6%)` | `google_cloud_console_api_key` | 否 | 按 Atlas 原始 `env.template` 的 Google Maps 条目在 Google Cloud Console 开通相应 API 并创建 key。 |

## 5. 查题与筛题

Atlas 的本地查询入口：

```bash
uv run python -m nexus.extensions.tasks.mcp_benchmark.atlas_selection summary
uv run python -m nexus.extensions.tasks.mcp_benchmark.atlas_selection api-counts --format table
uv run python -m nexus.extensions.tasks.mcp_benchmark.atlas_selection query --mode api-free --format table
uv run python -m nexus.extensions.tasks.mcp_benchmark.atlas_selection query --contains-servers github --format table
```

generator 内可用的 Atlas 筛题参数：

| 变量名 | 含义 |
| --- | --- |
| `MCP_BENCHMARK_ATLAS_MODE` | Atlas 用户模式，只能是 `api-free` 或 `original` |
| `MCP_BENCHMARK_ATLAS_CONTAINS_SERVERS` | 只保留包含指定 server 的题 |
| `MCP_BENCHMARK_ATLAS_CONTAINS_MATCH` | `all` 或 `any` |
| `MCP_BENCHMARK_ATLAS_ALLOWED_SERVERS` | 只保留其完整 server 集合是给定集合子集的题 |
| `MCP_BENCHMARK_ATLAS_ALLOWED_KEYED_SERVERS` | 只保留其“私有凭据依赖集合”是给定集合子集的题 |
| `MCP_BENCHMARK_ATLAS_ALLOWED_API_SERVERS` | `MCP_BENCHMARK_ATLAS_ALLOWED_KEYED_SERVERS` 的别名 |
| `MCP_BENCHMARK_CASE_IDS` | 在 Atlas 选择器之后再做精确 case_id 钉死 |
| `MCP_BENCHMARK_SKIP` / `MCP_BENCHMARK_LIMIT` | 分页和 smoke 用 |

几个常用例子：

```bash
# 看 api-free 现在能跑哪些题
uv run python -m nexus.extensions.tasks.mcp_benchmark.atlas_selection query \
  --mode api-free \
  --format table

# 只看包含 github 的题
uv run python -m nexus.extensions.tasks.mcp_benchmark.atlas_selection query \
  --contains-servers github \
  --contains-match any \
  --format table

# 只保留“私有凭据依赖集合”属于 {github,mongodb} 的题
uv run python -m nexus.extensions.tasks.mcp_benchmark.atlas_selection query \
  --allowed-keyed-servers github,mongodb \
  --format table

# 直接拿 case_id，后面喂给 MCP_BENCHMARK_CASE_IDS
uv run python -m nexus.extensions.tasks.mcp_benchmark.atlas_selection query \
  --contains-servers github \
  --allowed-keyed-servers github \
  --format case-ids
```

怎么理解这几个筛题参数：

- 你“只有少量 token”时，优先用 `MCP_BENCHMARK_ATLAS_ALLOWED_KEYED_SERVERS` 控制私有 API 族。
- 你想严格限制整道题所有 MCP 依赖时，再用 `MCP_BENCHMARK_ATLAS_ALLOWED_SERVERS`。
- 你只想挑出某类题，例如 GitHub 题，再叠加 `MCP_BENCHMARK_ATLAS_CONTAINS_SERVERS=github`。

## 6. `api-free` 全流程

`api-free` 的定位是：在缺少 original 私有凭据时，先把当前 benchmark 已经能 best-effort 跑通的 Atlas 子集跑起来。它不是“严格纯 mock”。

### 6.1 跑当前全部 `api-free` 可跑集

```bash
export MCP_BENCHMARK_ATLAS_MODE=api-free

unset MCP_BENCHMARK_ATLAS_CONTAINS_SERVERS
unset MCP_BENCHMARK_ATLAS_CONTAINS_MATCH
unset MCP_BENCHMARK_ATLAS_ALLOWED_SERVERS
unset MCP_BENCHMARK_ATLAS_ALLOWED_KEYED_SERVERS
unset MCP_BENCHMARK_ATLAS_ALLOWED_API_SERVERS
unset MCP_BENCHMARK_CASE_IDS
unset MCP_BENCHMARK_SKIP
unset MCP_BENCHMARK_LIMIT

export MCP_BENCHMARK_GENERATION_REPORT=workspace/mcp/tmp/generation_atlas_apifree.json

uv run nexus run-batch \
  -g nexus/extensions/tasks/mcp_benchmark/generator.py \
  -d workspace/mcp/tmp/batch_atlas_apifree.db \
  --max-concurrent 20 \
  --export-format full
```

### 6.2 跑一个 `api-free` 子集

```bash
export MCP_BENCHMARK_ATLAS_MODE=api-free
export MCP_BENCHMARK_ATLAS_ALLOWED_SERVERS=filesystem,git,mcp-code-executor,mcp-server-code-runner
export MCP_BENCHMARK_LIMIT=20

uv run nexus run-batch \
  -g nexus/extensions/tasks/mcp_benchmark/generator.py \
  -d workspace/mcp/tmp/batch_atlas_apifree_subset.db \
  --max-concurrent 20 \
  --export-format full
```

### 6.3 已验证的 `api-free` MongoDB smoke

这个 smoke 会刻意命中 `mongodb` 题，用来验证 Atlas `api-free` 最容易出问题的 service 路径。

```bash
export MCP_BENCHMARK_ATLAS_MODE=api-free
export MCP_BENCHMARK_ATLAS_CONTAINS_SERVERS=mongodb
unset MCP_BENCHMARK_ATLAS_CONTAINS_MATCH
unset MCP_BENCHMARK_ATLAS_ALLOWED_SERVERS
unset MCP_BENCHMARK_ATLAS_ALLOWED_KEYED_SERVERS
unset MCP_BENCHMARK_ATLAS_ALLOWED_API_SERVERS
unset MCP_BENCHMARK_CASE_IDS
unset MCP_BENCHMARK_SKIP
export MCP_BENCHMARK_LIMIT=1
export MCP_BENCHMARK_GENERATION_REPORT=workspace/mcp/tmp/generation_atlas_apifree_mongodb_smoke.json

uv run nexus run-batch \
  -g nexus/extensions/tasks/mcp_benchmark/generator.py \
  -d workspace/mcp/tmp/batch_atlas_apifree_mongodb_smoke.db \
  --max-concurrent 1 \
  --export-format full \
  --export-path workspace/mcp/tmp/export_atlas_apifree_mongodb_smoke
```

验证结果：

- `db`: `workspace/mcp/tmp/batch_atlas_apifree_mongodb_smoke.db`
- `run_id`: `atlas-mock-688ba1b3e95696e72dd93e87`
- `status`: `success`
- `passed`: `1`
- `score`: `0.642857142857143`
- `main_trajectory_len`: `19`

注：这里的 `run_id` 仍沿用历史 `atlas-mock-*` 命名，不影响当前文档里的用户模式名；对外统一称 `api-free`。

## 7. `original` 全流程

`original` 适合“你已经准备好了部分或全部私有 API 凭据”的场景。做法不是一上来硬跑 500 题，而是先按你手里已有的凭据族缩成子集，再批量跑。

### 7.1 跑未加额外筛选的 `original`

前提：你已经把上面 API 表里已有的凭据注入到了本地 benchmark 凭据配置里，字段名和表中一致。

```bash
export MCP_BENCHMARK_ATLAS_MODE=original

unset MCP_BENCHMARK_ATLAS_CONTAINS_SERVERS
unset MCP_BENCHMARK_ATLAS_CONTAINS_MATCH
unset MCP_BENCHMARK_ATLAS_ALLOWED_SERVERS
unset MCP_BENCHMARK_ATLAS_ALLOWED_KEYED_SERVERS
unset MCP_BENCHMARK_ATLAS_ALLOWED_API_SERVERS
unset MCP_BENCHMARK_CASE_IDS
unset MCP_BENCHMARK_SKIP
unset MCP_BENCHMARK_LIMIT

export MCP_BENCHMARK_GENERATION_REPORT=workspace/mcp/tmp/generation_atlas_original.json

uv run nexus run-batch \
  -g nexus/extensions/tasks/mcp_benchmark/generator.py \
  -d workspace/mcp/tmp/batch_atlas_original.db \
  --max-concurrent 20 \
  --export-format full
```

### 7.2 只跑某个 API 族的 `original` 子集

例子：如果你现在只有 GitHub 这一类私有凭据，就把题集限制成“包含 GitHub，且私有依赖只来自 GitHub”：

```bash
export MCP_BENCHMARK_ATLAS_MODE=original
export MCP_BENCHMARK_ATLAS_CONTAINS_SERVERS=github
export MCP_BENCHMARK_ATLAS_CONTAINS_MATCH=any
export MCP_BENCHMARK_ATLAS_ALLOWED_KEYED_SERVERS=github
export MCP_BENCHMARK_LIMIT=20

uv run nexus run-batch \
  -g nexus/extensions/tasks/mcp_benchmark/generator.py \
  -d workspace/mcp/tmp/batch_atlas_original_github_subset.db \
  --max-concurrent 20 \
  --export-format full
```

如果你有 GitHub + MongoDB：

```bash
export MCP_BENCHMARK_ATLAS_MODE=original
export MCP_BENCHMARK_ATLAS_ALLOWED_KEYED_SERVERS=github,mongodb

uv run nexus run-batch \
  -g nexus/extensions/tasks/mcp_benchmark/generator.py \
  -d workspace/mcp/tmp/batch_atlas_original_github_mongodb.db \
  --max-concurrent 20 \
  --export-format full
```

如果你还想精确钉死 case_id：

```bash
export MCP_BENCHMARK_CASE_IDS="$(
  uv run python -m nexus.extensions.tasks.mcp_benchmark.atlas_selection query \
    --contains-servers github \
    --allowed-keyed-servers github \
    --format case-ids
)"
```

### 7.3 已验证的 `original` smoke

```bash
export MCP_BENCHMARK_ATLAS_MODE=original
export MCP_BENCHMARK_ATLAS_ALLOWED_SERVERS=cli-mcp-server,git,github
export MCP_BENCHMARK_ATLAS_ALLOWED_KEYED_SERVERS=github
export MCP_BENCHMARK_LIMIT=1
export MCP_BENCHMARK_GENERATION_REPORT=workspace/mcp/tmp/generation_atlas_original_generator_smoke.json

uv run nexus run-batch \
  -g nexus/extensions/tasks/mcp_benchmark/generator.py \
  -d workspace/mcp/tmp/batch_atlas_original_generator_smoke.db \
  --max-concurrent 1 \
  --export-format full
```

验证结果：

- `db`: `workspace/mcp/tmp/batch_atlas_original_generator_smoke.db`
- `run_id`: `atlas-api-68940d8355faa0b720f0b485`
- `status`: `success`
- `passed`: `1`
- `score`: `1.0`

注：这里的 `run_id` 仍沿用历史 `atlas-api-*` 命名，不影响当前文档里的用户模式名；对外统一称 `original`。

这条 smoke 证明 `original` 这条链路已经在 Gongfeng 上完整打通，包括 runtime 启动、凭据注入、GitHub MCP wiring 和 judge。

## 8. 每题依赖矩阵

每一道题的依赖矩阵见：

- [atlas_api_matrix.md](atlas_api_matrix.md)

重生成命令：

```bash
uv run python -m nexus.extensions.tasks.mcp_benchmark.atlas_selection matrix \
  --output nexus/extensions/tasks/mcp_benchmark/docs/atlas_api_matrix.md
```

矩阵里最值得看的几列：

- `依赖 MCP 服务`
- `需要私有凭据的服务`
- `凭据字段名`
- `api-free 可跑`

如果你只想看 API 来源统计而不是逐题矩阵：

```bash
uv run python -m nexus.extensions.tasks.mcp_benchmark.atlas_selection api-counts --format table
```
