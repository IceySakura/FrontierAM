# MCP Benchmark Toolathlon（Gongfeng）

相关文档：

- [总览](../README.md)
- [Atlas 详情](atlas.md)
- [Universe 详情](universe.md)

## 0. 数据准备

Toolathlon 现在先走 prepare，再走筛题 / generator：

```bash
uv run python -m nexus.extensions.tasks.mcp_benchmark.prepare_data --datasets toolathlon
```

如果本地 `workspace/mcp/datasets/toolathlon` 已经准备好，只想重建 manifest：

```bash
uv run python -m nexus.extensions.tasks.mcp_benchmark.prepare_data --datasets toolathlon --skip-sync
```

- 上游题目来源：`workspace/mcp/datasets/toolathlon/tasks/finalpool/*/task_config.json` + `docs/task.md`
- prepare 产物：`workspace/mcp/converted/toolathlon.jsonl`
- 后续 `toolathlon_selection.py`、`datasets.py`、`generator.py` 都只读取这个 JSONL manifest，不再直接扫描题目目录生成 case 列表

## 1. 两种运行模式

- `api-free`：面向“没有 original 额外凭据，或者只想先低成本跑通一批题”时的 best effort 路径。它会优先走当前 benchmark 已经接好的 mock / shim，但不是“完全离线”模式。
- `original`：按 Toolathlon 原始 MCP server / API 链路运行。对 Toolathlon 来说，这条链路不只包含远端 API，还混有一批需要完整上游本地 app 部署的题，因此在 Gongfeng 上更推荐始终按 API / server 子集运行，而不是直接放开整池。

当前 Toolathlon 快照来自：

```bash
uv run python -m nexus.extensions.tasks.mcp_benchmark.toolathlon_selection summary
```

当前结果：

- Toolathlon 总题量：`108`
- `api-free` 可跑：`86`
- `api-free` 当前的主要缺口：
  - `google-cloud`：`8`
  - `github`：`7`
  - `huggingface`：`5`
  - `wandb`：`3`

为什么 `api-free` 不是 `100`：

- 当前 benchmark 的 Toolathlon mock 路径已经覆盖了 Google Workspace、Notion、Snowflake、Google Maps、YouTube、Canvas、WooCommerce 等多条链路。
- 但 `google-cloud` 这 `8` 题当前仍未接入。
- 同时 `github`、`huggingface`、`wandb` 在 `api-free` 下仍会 fallback 到 live 服务，因此在“没有这些 original 凭据”的前提下必须排除。

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
export MCP_BENCHMARK_DATASETS=toolathlon

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

## 3.5 凭据文件方式（组内复现推荐）

第 3 节给的环境变量是最小基线。 在组内已经准备好 Toolathlon 全套凭据的场景下，更推荐直接指向 **凭据文件目录** ，而不是逐个 token export。 这也是当前组内跑批实际使用的方式。

需要的三样东西：

| 环境变量 | 指向 | 用途 |
| --- | --- | --- |
| `NEXUS_MCP_BENCHMARK_TOOLATHLON_ROOT` | 本地 Toolathlon 源码目录 | manifest / preprocess / evaluation 脚本来源 |
| `MCP_BENCHMARK_CREDENTIALS_KEYS_PY` | `keys.py` 文件 | 所有 API token（GitHub / Notion / HuggingFace / WandB / OpenAI 等）集中定义点 |
| `MCP_BENCHMARK_CREDENTIAL_BUNDLE_PATH` | `credential_fils/` 目录 | 文件类凭据：`google_credentials.json` 、 Snowflake RSA key 、 Notion OAuth session 等 |

加载路径：容器启动时 env var → `load_workspace_keys()` → `template_values()` → MCP config 里的 `${token.xxx}` 占位符被逐一替换。

凭据内容怎么准备参考上游 Toolathlon 的 [`README.md`](../../../../../workspace/mcp/datasets/toolathlon/README.md) 和 [`how2register_accounts.md`](../../../../../workspace/mcp/datasets/toolathlon/global_preparation/how2register_accounts.md) 。

### 跑单题示例（组内复现模板）

```bash
export NEXUS_MCP_BENCHMARK_TOOLATHLON_ROOT=/path/to/Toolathlon
export MCP_BENCHMARK_CREDENTIALS_KEYS_PY=/path/to/keys.py
export MCP_BENCHMARK_CREDENTIAL_BUNDLE_PATH=/path/to/credential_fils

export MCP_BENCHMARK_DATASETS=toolathlon
export MCP_BENCHMARK_TOOLATHLON_MODE=original
export MCP_BENCHMARK_CASE_IDS=flagged-transactions
export MCP_BENCHMARK_LIMIT=1

# 第 3 节的 runtime / LLM 基线变量照贴即可

uv run nexus run-batch \
  -g nexus/extensions/tasks/mcp_benchmark/generator.py \
  -d outputs/batch_single.db \
  --max-concurrent 1 \
  --export-format full
```

所有 `export` 也可以写进一个 `.env.sh` ，用 `source .env.sh` 一键加载。

## 4. Toolathlon 需要哪些额外凭据 / 配置

下面这张表按 Toolathlon 每道题的 `key_required_servers` 统计；一道题命中多个服务时会重复计数。

如需重算：

```bash
uv run python -m nexus.extensions.tasks.mcp_benchmark.toolathlon_selection api-counts --format table
```

上游原始说明优先参考：

- [Toolathlon `README.md`](../../../../../workspace/mcp/datasets/toolathlon/README.md)
- [Toolathlon `how2register_accounts.md`](../../../../../workspace/mcp/datasets/toolathlon/global_preparation/how2register_accounts.md)
- [Toolathlon `token_key_session_example.py`](../../../../../workspace/mcp/datasets/toolathlon/configs/token_key_session_example.py)

| 服务来源 | Toolathlon 题量 | 需要准备的字段 / 工件 | 获取方式 / 备注 |
| --- | --- | --- | --- |
| Google Sheets | `11 (10.2%)` | `google_oauth_credentials_file` | 按 Toolathlon 上游 Google Account 部分执行 `automated_google_setup.sh`，生成 `google_credentials.json`。这一份 Google OAuth 凭据文件同时服务 `google_sheet`、`google_calendar`、`google_forms`。 |
| Google Cloud | `8 (7.4%)` | `gcp_project_id`, `gcp_service_account_path` | 按上游 Google Account 部分准备 GCP project 和 service account。注意：当前 benchmark 还没有打通 `google-cloud` 题目，这 `8` 题暂时不能在 Gongfeng 上跑。 |
| Notion | `8 (7.4%)` | `notion_integration_key`, `notion_integration_key_eval`, `source_notion_page_url`, `eval_notion_page_url` | 按上游 Notion Account 部分复制 source/eval 页面，并创建 source/eval 两套 integration secret。 |
| GitHub | `7 (6.5%)` | `github_token` | 按上游 `automated_additional_services.sh` 或 GitHub PAT 流程准备 token。 |
| Google Maps | `6 (5.6%)` | `google_cloud_console_api_key` | 同样走上游 Google Account 配置，在 Google Cloud Console 里为 Maps 相关 API 创建 key。 |
| Hugging Face | `5 (4.6%)` | `huggingface_token` | 按上游 `automated_additional_services.sh` 或 Hugging Face token 流程准备。 |
| Snowflake | `4 (3.7%)` | `snowflake_account`, `snowflake_role`, `snowflake_user`, `snowflake_private_key_file` | 按上游 Snowflake Account 部分创建账号，并生成 `snowflake_rsa_key.p8` / 公钥对。 |
| WandB | `3 (2.8%)` | `wandb_api_key` | 按上游 `automated_additional_services.sh` 或 WandB token 流程准备。 |
| Google Calendar | `2 (1.9%)` | `google_oauth_credentials_file` | 与 `google_sheet` 共用同一份 Google OAuth 凭据文件。 |
| Google Forms | `2 (1.9%)` | `google_oauth_credentials_file` | 与 `google_sheet` 共用同一份 Google OAuth 凭据文件。 |
| YouTube | `2 (1.9%)` | `google_cloud_console_api_key` | 与 `google_map` 共用同一份 Google Cloud Console API key。 |

额外说明：

- Canvas、WooCommerce、Emails、K8s 这类 Toolathlon 本地 app / 本地账号链路，主要由任务自带 override 和上游本地部署脚本提供，不单独记入这张“额外凭据 / 配置”表。
- 如果你要跑依赖这些本地 app 的 `original` 题，仍然需要完整参考 Toolathlon 上游的 `deploy_containers.sh` 和相关部署说明。

## 5. 查题与筛题

Toolathlon 的本地查询入口：

```bash
uv run python -m nexus.extensions.tasks.mcp_benchmark.toolathlon_selection summary
uv run python -m nexus.extensions.tasks.mcp_benchmark.toolathlon_selection api-counts --format table
uv run python -m nexus.extensions.tasks.mcp_benchmark.toolathlon_selection query --mode api-free --format table
uv run python -m nexus.extensions.tasks.mcp_benchmark.toolathlon_selection query --contains-servers github --format table
```

generator 内可用的 Toolathlon 筛题参数：

| 变量名 | 含义 |
| --- | --- |
| `MCP_BENCHMARK_TOOLATHLON_MODE` | Toolathlon 用户模式，只能是 `api-free` 或 `original` |
| `MCP_BENCHMARK_TOOLATHLON_CONTAINS_SERVERS` | 只保留包含指定 server 的题 |
| `MCP_BENCHMARK_TOOLATHLON_CONTAINS_MATCH` | `all` 或 `any` |
| `MCP_BENCHMARK_TOOLATHLON_ALLOWED_SERVERS` | 只保留其完整 server 集合是给定集合子集的题 |
| `MCP_BENCHMARK_TOOLATHLON_ALLOWED_KEYED_SERVERS` | 只保留其“额外凭据依赖集合”是给定集合子集的题 |
| `MCP_BENCHMARK_TOOLATHLON_ALLOWED_API_SERVERS` | `MCP_BENCHMARK_TOOLATHLON_ALLOWED_KEYED_SERVERS` 的别名 |
| `MCP_BENCHMARK_CASE_IDS` | 在 Toolathlon 选择器之后再做精确 case_id 钉死 |
| `MCP_BENCHMARK_SKIP` / `MCP_BENCHMARK_LIMIT` | 分页和 smoke 用 |

几个常用例子：

```bash
# 看 api-free 现在能跑哪些题
uv run python -m nexus.extensions.tasks.mcp_benchmark.toolathlon_selection query \
  --mode api-free \
  --format table

# 只看包含 github 的题
uv run python -m nexus.extensions.tasks.mcp_benchmark.toolathlon_selection query \
  --contains-servers github \
  --contains-match any \
  --format table

# 只保留“额外凭据依赖集合”属于 {github,notion} 的题
uv run python -m nexus.extensions.tasks.mcp_benchmark.toolathlon_selection query \
  --allowed-keyed-servers github,notion \
  --format table

# 直接拿 case_id，后面喂给 MCP_BENCHMARK_CASE_IDS
uv run python -m nexus.extensions.tasks.mcp_benchmark.toolathlon_selection query \
  --contains-servers github \
  --allowed-keyed-servers github \
  --format case-ids
```

怎么理解这几个筛题参数：

- 你“只有少量 token”时，优先用 `MCP_BENCHMARK_TOOLATHLON_ALLOWED_KEYED_SERVERS` 控制远端 API / 凭据族。
- 你想把题目依赖的 MCP server 也一起收紧时，再用 `MCP_BENCHMARK_TOOLATHLON_ALLOWED_SERVERS`。
- 你只想挑某类 server 题，例如 GitHub 题，再叠加 `MCP_BENCHMARK_TOOLATHLON_CONTAINS_SERVERS=github`。

## 6. `api-free` 全流程

`api-free` 的定位是：在缺少 original 额外凭据时，先把当前 benchmark 已经能 best-effort 跑通的 Toolathlon 子集跑起来。它不是“严格纯 mock”。

### 6.1 跑当前全部 `api-free` 可跑集

```bash
export MCP_BENCHMARK_DATASETS=toolathlon
export MCP_BENCHMARK_TOOLATHLON_MODE=api-free

unset MCP_BENCHMARK_TOOLATHLON_CONTAINS_SERVERS
unset MCP_BENCHMARK_TOOLATHLON_CONTAINS_MATCH
unset MCP_BENCHMARK_TOOLATHLON_ALLOWED_SERVERS
unset MCP_BENCHMARK_TOOLATHLON_ALLOWED_KEYED_SERVERS
unset MCP_BENCHMARK_TOOLATHLON_ALLOWED_API_SERVERS
unset MCP_BENCHMARK_CASE_IDS
unset MCP_BENCHMARK_SKIP
unset MCP_BENCHMARK_LIMIT

export MCP_BENCHMARK_GENERATION_REPORT=workspace/mcp/tmp/generation_toolathlon_apifree.json

uv run nexus run-batch \
  -g nexus/extensions/tasks/mcp_benchmark/generator.py \
  -d workspace/mcp/tmp/batch_toolathlon_apifree.db \
  --max-concurrent 20 \
  --export-format full
```

### 6.2 跑一个 `api-free` 子集

```bash
export MCP_BENCHMARK_DATASETS=toolathlon
export MCP_BENCHMARK_TOOLATHLON_MODE=api-free
export MCP_BENCHMARK_TOOLATHLON_ALLOWED_SERVERS=filesystem,terminal
export MCP_BENCHMARK_LIMIT=20

uv run nexus run-batch \
  -g nexus/extensions/tasks/mcp_benchmark/generator.py \
  -d workspace/mcp/tmp/batch_toolathlon_apifree_subset.db \
  --max-concurrent 20 \
  --export-format full
```

### 6.3 已验证的 `api-free` smoke

```bash
export MCP_BENCHMARK_DATASETS=toolathlon
export MCP_BENCHMARK_TOOLATHLON_MODE=api-free
export MCP_BENCHMARK_TOOLATHLON_ALLOWED_SERVERS=filesystem,terminal
export MCP_BENCHMARK_CASE_IDS=privacy-desensitization
export MCP_BENCHMARK_LIMIT=1
export MCP_BENCHMARK_GENERATION_REPORT=workspace/mcp/tmp/generation_toolathlon_apifree_verify.json

uv run nexus run-batch \
  -g nexus/extensions/tasks/mcp_benchmark/generator.py \
  -d workspace/mcp/tmp/batch_toolathlon_apifree_verify.db \
  --max-concurrent 1 \
  --export-format full
```

验证结果：

- `db`: `workspace/mcp/tmp/batch_toolathlon_apifree_verify.db`
- `run_id`: `toolathlon-mock-privacy-desensitization`
- `status`: `success`
- `passed`: `0`
- `score`: `0.0`

注：这里的 `run_id` 会沿用内部 `toolathlon-mock-*` server_mode 命名；对外用户模式仍统一称 `api-free`。
注：这里的 smoke 主要用于验证 generator / runtime / MCP / evaluator 链路可执行；`passed` / `score` 仍取决于当前 agent 在该样例上的表现。

## 7. `original` 全流程

Toolathlon 的 `original` 在 Gongfeng 上更适合“按 API / server 家族缩成子集后再跑”。如果你没有完整补齐 Toolathlon 上游的本地 app 部署、Google / Notion / Snowflake 等配置，不建议直接放开大池。

### 7.1 只跑 GitHub 子集

```bash
export MCP_BENCHMARK_DATASETS=toolathlon
export MCP_BENCHMARK_TOOLATHLON_MODE=original
export MCP_BENCHMARK_TOOLATHLON_CONTAINS_SERVERS=github
export MCP_BENCHMARK_TOOLATHLON_CONTAINS_MATCH=any
export MCP_BENCHMARK_TOOLATHLON_ALLOWED_KEYED_SERVERS=github
export MCP_BENCHMARK_LIMIT=20

uv run nexus run-batch \
  -g nexus/extensions/tasks/mcp_benchmark/generator.py \
  -d workspace/mcp/tmp/batch_toolathlon_original_github_subset.db \
  --max-concurrent 20 \
  --export-format full
```

### 7.2 只跑 Notion + Google OAuth 子集

```bash
export MCP_BENCHMARK_DATASETS=toolathlon
export MCP_BENCHMARK_TOOLATHLON_MODE=original
export MCP_BENCHMARK_TOOLATHLON_ALLOWED_KEYED_SERVERS=notion,google_sheet,google_calendar,google_forms

uv run nexus run-batch \
  -g nexus/extensions/tasks/mcp_benchmark/generator.py \
  -d workspace/mcp/tmp/batch_toolathlon_original_notion_google.db \
  --max-concurrent 20 \
  --export-format full
```

### 7.3 用 case_id 精确钉死

```bash
export MCP_BENCHMARK_CASE_IDS="$(
  uv run python -m nexus.extensions.tasks.mcp_benchmark.toolathlon_selection query \
    --contains-servers github \
    --allowed-keyed-servers github \
    --format case-ids
)"
```

### 7.4 已验证的 `original` smoke

```bash
export MCP_BENCHMARK_DATASETS=toolathlon
export MCP_BENCHMARK_TOOLATHLON_MODE=original
export MCP_BENCHMARK_TOOLATHLON_CONTAINS_SERVERS=github
export MCP_BENCHMARK_TOOLATHLON_CONTAINS_MATCH=any
export MCP_BENCHMARK_TOOLATHLON_ALLOWED_KEYED_SERVERS=github
export MCP_BENCHMARK_CASE_IDS=sync-todo-to-readme
export MCP_BENCHMARK_LIMIT=1
export MCP_BENCHMARK_GENERATION_REPORT=workspace/mcp/tmp/generation_toolathlon_original_verify.json

uv run nexus run-batch \
  -g nexus/extensions/tasks/mcp_benchmark/generator.py \
  -d workspace/mcp/tmp/batch_toolathlon_original_verify.db \
  --max-concurrent 1 \
  --export-format full
```

验证结果：

- `db`: `workspace/mcp/tmp/batch_toolathlon_original_verify.db`
- `run_id`: `toolathlon-api-sync-todo-to-readme`
- `status`: `success`
- `passed`: `0`
- `score`: `0.0`

注：这里的 `run_id` 会沿用内部 `toolathlon-api-*` server_mode 命名；对外用户模式仍统一称 `original`。
注：这里的 smoke 主要用于验证 generator / runtime / MCP / evaluator 链路可执行；`passed` / `score` 仍取决于当前 agent 在该样例上的表现。

## 8. 每题依赖矩阵

每一道题的依赖矩阵见：

- [toolathlon_api_matrix.md](toolathlon_api_matrix.md)

重生成命令：

```bash
uv run python -m nexus.extensions.tasks.mcp_benchmark.toolathlon_selection matrix \
  --output nexus/extensions/tasks/mcp_benchmark/docs/toolathlon_api_matrix.md
```

矩阵里最值得看的几列：

- `依赖 MCP 服务`
- `需要额外凭据的服务`
- `凭据字段名`
- `api-free 可跑`

如果你只想看 API / 凭据来源统计而不是逐题矩阵：

```bash
uv run python -m nexus.extensions.tasks.mcp_benchmark.toolathlon_selection api-counts --format table
```
