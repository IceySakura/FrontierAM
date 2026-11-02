---
title: 环境变量参考
audience: 需要配 aidata 或调优行为的开发者
prerequisites: 无
related:
  - ./cli.md
  - ../how-to/troubleshoot.md
---

# 环境变量参考

按分组列出 。 未在此处提到但代码里搜到的变量多半是遗留 / 未启用 —— 不要依赖它们 。

## 上游凭据 —— 必填

| 变量 | 作用 |
|---|---|
| `GONGFENG_API_BASE_URL` | Gongfeng sandbox provider 地址 。 例 ：`http://hyrl-sandbox.prod.woa.com:8052` 。 |
| `GONGFENG_TOKEN` | Gongfeng token ；从团队 wiki 拿 。 |
| `SEARCH_OPENAI_BASE_URL` | agent + mock LLM 用的 LLM 网关 。 例 ：`http://llm-api.model-eval.woa.com` 。 |
| `SEARCH_OPENAI_API_KEY` | 网关 key ，格式 `<APP_ID>:<APP_KEY>` 。 |
| `SEARCH_OPENAI_MODEL` | 模型 ID ，如 `api_naci_default_gemini-3.1-pro-preview` 。 |

Orchestrator 内部按模型族切模板与凭据 —— `MODELS` 表已经打包 base_url + model + 可选
`SEARCH_OPENAI_API_KEY` ，运行时无需手工改 `.env` 。 手工 `aidata.py run` 走 `.env` 里的默认值 。

## Judge 网关（可选 ；缺省复用 `SEARCH_OPENAI_*` ）

用于 `judge_claim` primitive 。 若不显式配 ，会 fallback 到 agent 的 `SEARCH_OPENAI_*` 。

| 变量 | 作用 |
|---|---|
| `JUDGE_OPENAI_BASE_URL` | judge LLM 网关地址 。 |
| `JUDGE_OPENAI_API_KEY` | judge 网关 key 。 |
| `JUDGE_OPENAI_MODEL` | judge 模型 ID —— 允许与 agent 用不同 model 。 |
| `NEXUS_JUDGE_BASE_URL` / `NEXUS_JUDGE_API_KEY` / `NEXUS_JUDGE_MODEL` | 别名前缀 ，与 `JUDGE_OPENAI_*` 语义相同 。 |
| `NEXUS_JUDGE_MAX_RETRIES` | UNAVAILABLE 时的重试次数 ，默认 5 。 orchestrator 通过 `--judge-retries` 写入 。 |

## AIDATA 专属 —— case 与 sandbox

| 变量 | 作用 | 默认 |
|---|---|---|
| `AIDATA_CASES_ROOT` | cases 树根目录 。 CLI 、generator 、tests conftest 三方共用 。 未设 → `<aidata-package>/cases/` 。 | 未设 |
| `AIDATA_CASE_IDS` | 逗号分隔 case_id allowlist（`generator.py` 用） 。 空 = 全量 。 | 未设 |
| `AIDATA_SKIP_PENDING_UPSTREAM` | `1`（默认） 跳过声明了 `pending_upstream:` 的 case ；`0` 强制带上 。 | `1` |
| `AIDATA_CONFLICT_GROUPS` | 逗号分隔 mutex group ；写到每个 BatchItem 的 metadata 。 | 未设 |
| `AIDATA_MAIN_IMAGE` / `NEXUS_AIDATA_MAIN_IMAGE` | 覆盖 main runtime 镜像 。 | 未设 |
| `AIDATA_CONFINE_MODE` | `enforce` = 缺失套件转硬性中止 ；`warn` = 打印警告放行 ；`off` = 禁用 confine 。 | `enforce` |
| `AIDATA_CONFINE_NET_DENY` | 覆盖默认 confine 网络 deny list 。 | 内置默认 |

## AIDATA 专属 —— rubric 打分

| 变量 | 作用 | 默认 |
|---|---|---|
| `AIDATA_STATE_DUMP` | `0` 关掉 state dump ；其它值启用 。 | `1` |
| `AIDATA_STATE_DUMP_DIR` | state dump 落盘目录 ；orchestrator 会 per-run 覆盖 。 | `/tmp/aidata-state/` |
| `AIDATA_SCORE_THRESHOLD` | ScorePolicy 阈值截断点 。 | `1.0` |
| `AIDATA_SCORE_CONTINUOUS` | ScorePolicy 是否输出连续分数 。 | `True` |
| `AIDATA_SCORE_USE_THRESHOLD` | ScorePolicy 是否应用截断 。 | `False` |
| `NEXUS_AIDATA_RUBRIC_LEGACY` | `1` 走 Fetcher 门面之前的直接 state 访问路径（回滚逃生阀） 。 | 未设 |

`AIDATA_SCORE_*` 三兄弟组合语义见
[`../explanation/rubric-scoring-model.md`](../explanation/rubric-scoring-model.md) 。 优先级 ：
env > `rubric['config']['score']` > 默认值 。 代码入口 ：
`nexus/extensions/tasks/aidata/rubric/policy.py::ScorePolicy` 。

## AIDATA 专属 —— image layout 覆盖

`engine/paths.py::_IMAGE_LAYOUTS` 按 image 前缀选 MCP_PY / SHELL_PATH 。 这三个变量允许对
非典型镜像临时覆盖 ：

| 变量 | 作用 |
|---|---|
| `NEXUS_AIDATA_MCP_PY` | 覆盖 MCP python 解释器路径 。 |
| `NEXUS_AIDATA_SHELL_PATH` | 覆盖 shell MCP 用的 shell 路径 。 |
| `NEXUS_AIDATA_PDF_TOOLS_BIN` | 覆盖 pdf-tools MCP server 二进制路径 。 |

## Mock LLM 旋钮

| 变量 | 作用 | 默认 |
|---|---|---|
| `NEXUS_MOCK_LLM_MAX_CALLS` | 单进程内对 mock LLM 的调用上限 。 | `50` |
| `NEXUS_MOCK_SEED` | 复现性 seed 。 | 取 `hash(user_prompt)` |

## Rubric 并发

| 变量 | 作用 | 默认 |
|---|---|---|
| `NEXUS_RUBRIC_CONCURRENCY` | 单个 rubric 内并行评估 check 的上限 。 传入 `1` 即串行 。 | `8` |

## MockServer 数据目录（extra_mcp_servers 用）

| 变量 | 作用 |
|---|---|
| `MOCK_SERVER_DATA_DIR` | MockServer 仓的 mock 需要 per-run 隔离的落盘目录 。 |
| `MOCK_SERVER_NOTES_SEED` | notes mock 的初始 state 注入 。 |

## Viz

| 变量 | 作用 | 默认 |
|---|---|---|
| `NEXUS_VIZ_MAX_CONCURRENCY` | Viz Playground 并发跑数上限 。 | `8` |

## 加载优先级

`.env` 加载顺序（`aidata.py` / `aidata_orchestrator.py` / generator / tests conftest 一致） ：

1. `<AIDATA_CASES_ROOT>/../.env` —— 若存在 ，优先 。
2. `<repo-root>/.env` —— 上游 nexus 仓根 。

进程环境变量 **总是** 高于 `.env` 里的同名值 —— `export AIDATA_CASES_ROOT=...` 之后再跑就是那个值 。

## 相关文档

- CLI 参数完整表 —— [cli.md](./cli.md) 。
- 排障 —— [`../how-to/troubleshoot.md`](../how-to/troubleshoot.md) 。
- ScorePolicy 源码 —— `nexus/extensions/tasks/aidata/rubric/policy.py` 。
