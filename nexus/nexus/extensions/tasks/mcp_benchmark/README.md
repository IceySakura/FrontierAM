# MCP Benchmark 文档总览

本目录文档分两层：

- `README.md`：只保留运行模式、数据集规模、按 API 类别区分的使用情况。
- `docs/`：包含 `atlas.md`、`toolathlon.md`、`universe.md` 三份数据集说明，以及对应的 `*_api_matrix.md` 依赖矩阵。

## 文档入口

- [Atlas 详情](docs/atlas.md)
- [Toolathlon 详情](docs/toolathlon.md)
- [Universe 详情](docs/universe.md)

## 数据准备流程

当前 benchmark 改成两阶段：

1. 先执行 prepare 脚本，下载 / 更新上游数据仓库，并把题目转换成按数据集拆分的 JSONL manifest。
2. 再由 `generator.py`、各数据集选择器和 `matrix_report.py` 只读取这些 JSONL manifest 生成配置或报表。

准备命令：

```bash
uv run python -m nexus.extensions.tasks.mcp_benchmark.prepare_data
```

如果本地 `workspace/mcp/datasets/{toolathlon,atlas,universe}` 已经准备好，只想重建 manifest，可以跳过仓库同步：

```bash
uv run python -m nexus.extensions.tasks.mcp_benchmark.prepare_data --skip-sync
```

prepare 产物与来源：

| 数据集 | 上游来源 | 运行时题目来源 JSONL | 原始题目来源 |
| --- | --- | --- | --- |
| Toolathlon | `workspace/mcp/datasets/toolathlon` | `workspace/mcp/converted/toolathlon.jsonl` | `tasks/finalpool/*/task_config.json` + `docs/task.md` |
| Atlas | `workspace/mcp/datasets/atlas` + Hugging Face parquet cache | `workspace/mcp/converted/atlas.jsonl` | `ScaleAI/MCP-Atlas` 的 `MCP-Atlas.parquet` |
| Universe | `workspace/mcp/datasets/universe` | `workspace/mcp/converted/universe.jsonl` | `mcpuniverse/benchmark/configs/mcpuniverse/**/*.json` |

当前这三份 converted manifest 也已同步到共享目录：

- `/apdcephfs_gy4/share_303214677/tarschen/data/mcp/converted`

如果你不想在本地重新 prepare，可以直接从这个目录下载 `toolathlon.jsonl`、`atlas.jsonl`、`universe.jsonl`，或者直接把 benchmark 指到这个共享目录：

```bash
export NEXUS_MCP_BENCHMARK_CONVERTED_ROOT=/apdcephfs_gy4/share_303214677/tarschen/data/mcp/converted
```

Atlas 的 parquet 会在 prepare 阶段按现有 loader 自动缓存到 `workspace/mcp/datasets/atlas/hf_cache/MCP-Atlas.parquet`，除非你显式覆盖 `NEXUS_ATLAS_PARQUET_URI` / `NEXUS_ATLAS_PARQUET_CACHE_PATH`。

## 运行模式

当前只保留 `original` 模式：按各数据集原始 MCP server / API 链路运行。 只有在相关私有凭据、本地服务或上游依赖准备齐全时，题目才会被选入并正常执行。

## 数据集概览

| 数据集 | 总题量 | 详情 |
| --- | --- | --- |
| Atlas | `500` | [Atlas 详情](docs/atlas.md) |
| Toolathlon | `108` | [Toolathlon 详情](docs/toolathlon.md) |
| Universe | `232` | [Universe 详情](docs/universe.md) |

## 按 API 类别区分的使用情况

统计口径：按每道题的 `key_required_servers` 或额外凭据依赖统计；一道题命中多个 API / 服务时会重复计数。

### Atlas

| API 类别 | 题量 | 占比 | 凭据字段名 |
| --- | --- | --- | --- |
| `github` | `159` | `31.8%` | `github_token` |
| `oxylabs` | `94` | `18.8%` | `oxylabs_username`、`oxylabs_password` |
| `airtable` | `92` | `18.4%` | `airtable_api_key` |
| `weather-data` | `86` | `17.2%` | `weather_api_key` |
| `mongodb` | `79` | `15.8%` | `mongodb_connection_string` |
| `exa` | `77` | `15.4%` | `exa_api_key` |
| `national-parks` | `76` | `15.2%` | `nps_api_key` |
| `alchemy` | `69` | `13.8%` | `alchemy_api_key` |
| `notion` | `67` | `13.4%` | `notion_integration_key_eval`、`notion_integration_key` |
| `brave-search` | `63` | `12.6%` | `brave_api_key` |
| `e2b-server` | `61` | `12.2%` | `e2b_api_key` |
| `slack` | `60` | `12.0%` | `slack_mcp_xoxc_token`、`slack_mcp_xoxd_token` |
| `twelvedata` | `60` | `12.0%` | `twelve_data_api_key` |
| `lara-translate` | `53` | `10.6%` | `lara_access_key_id`、`lara_access_key_secret` |
| `google-workspace` | `50` | `10.0%` | `google_client_id`、`google_client_secret`、`google_refresh_token` |
| `google-maps` | `38` | `7.6%` | `google_cloud_console_api_key` |

### Toolathlon

| API 类别 | 题量 | 占比 | 凭据字段名 |
| --- | --- | --- | --- |
| `google_sheet` | `11` | `10.2%` | `google_oauth_credentials_file` |
| `google-cloud` | `8` | `7.4%` | `gcp_project_id`、`gcp_service_account_path` |
| `notion` | `8` | `7.4%` | `notion_integration_key`、`notion_integration_key_eval`、`source_notion_page_url`、`eval_notion_page_url` |
| `github` | `7` | `6.5%` | `github_token` |
| `google_map` | `6` | `5.6%` | `google_cloud_console_api_key` |
| `huggingface` | `5` | `4.6%` | `huggingface_token` |
| `snowflake` | `4` | `3.7%` | `snowflake_account`、`snowflake_role`、`snowflake_user`、`snowflake_private_key_file` |
| `wandb` | `3` | `2.8%` | `wandb_api_key` |
| `google_calendar` | `2` | `1.9%` | `google_oauth_credentials_file` |
| `google_forms` | `2` | `1.9%` | `google_oauth_credentials_file` |
| `youtube` | `2` | `1.9%` | `google_cloud_console_api_key` |

### Universe

| API 类别 | 题量 | 占比 | 凭据字段名 |
| --- | --- | --- | --- |
| `google-search` | `55` | `23.7%` | `serper_api_key` |
| `google-maps` | `45` | `19.4%` | `google_cloud_console_api_key` |
| `github` | `33` | `14.2%` | `github_token` |
| `notion` | `10` | `4.3%` | `notion_integration_key`、`notion_integration_key_eval`、`source_notion_page_url`、`eval_notion_page_url` |

更细的安装、筛题、运行示例和 smoke 结果见各数据集文档。
