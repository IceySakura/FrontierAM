# SWE-Pro + Codex + GPT-5.6-sol (AIHub 2.0 Responses)

Publishable example: Codex CLI on SWE-Bench Pro (731) with GPT-5.6-sol encrypted
reasoning on AIHub 2.0 `standard/v1`.

## Pinned versions

| knob | value |
|---|---|
| Codex | **`0.154.0` hard pin**（generator 不读 `SWE_CODEX_VERSION` / `CODEX_VERSION`） |
| Bus label（SFT 侧，非本 repo 维护） | `codex@0.154.0` |
| model | `gpt-5.6-sol` (override `SWE_MODEL`) |
| model catalog | `model_catalog.json` (0.154 `gpt-5.6-sol`, `tool_mode=direct`) |

Pin the **semver in this generator**, not `@latest` / not ambient env.

## Pinned wire

| knob | location | value |
|---|---|---|
| endpoint | client | `http://api.aihub.woa.com/standard/v1/responses` |
| api_key | client | bare `OPENAI_API_KEY` (no `?provider=` on `standard/v1`) |
| model | body | `gpt-5.6-sol` |
| timeout | client | `12000` |
| max_retries | client | `32` |
| cache_task_id | client_args | session `uuid4` (region pin) |
| store | body | `false` |
| include | body | `["reasoning.encrypted_content"]` |
| reasoning.effort | body | `max` (override `SWE_REASONING_EFFORT`) |
| reasoning.summary | body | `detailed` |
| reasoning.mode | body | `standard` (override `SWE_REASONING_MODE` / `GPT_REASONING_MODE`) |
| max_output_tokens | body | `128000` |
| parallel_tool_calls | body | `false` |
| prompt_cache_key | body | session `uuid4` |
| `capture_raw_io` | client_args | `true`（把 Responses 原包记进 `provider_call`，便于对密文；关了 Langfuse 轨迹还在） |
| Codex `use_azure_provider` | agent | **`false`** |
| **`wrap_plaintext_reasoning_content`** | agent | **`false`（必开）** |
| `model_context_window` | agent | `1000000` |
| `auto_compact_token_limit` | agent | `900000` |
| `extra_config_toml` | agent | `remote_compaction_v2 = false` (`SWE_CODEX_EXTRA_CONFIG_TOML`) |
| `extra_cli_args` | agent | `-c web_search="disabled"` |

`wrap_plaintext_reasoning_content=false` is the GPT pin. Default Codex/main wraps
`reasoning_content` and mints a local `rs_*` when an item has no id — Polarist /
AIHub `store=false` then rejects the next turn (`Encrypted content item_id did
not match` / `rs_ not found`). This generator turns wrap **off** so the converter
keeps native `reasoning_items` as-is: real `rs_` + `gAAAAA` only, no invented
ids, encrypted blobs without a provider id are dropped. Do not omit this field
and do not set it `true` for GPT-5.6 SWE-Pro. HY3/GLM plaintext wrap stays on
the Codex default (`true`) and is a different task.

Override endpoint with `SWE_OPENAI_RESPONSES_ENDPOINT` if you need legacy Polarist
(`APP_ID:APP_KEY?provider=…` on that path only).

## Anti-hack (default on)

Same four switches as `swepro-cc-gpt56`. Set any to `0`/`false` to disable.

## Langfuse

On by default. Needs `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` /
`LANGFUSE_BASE_URL` (or `LANGFUSE_HOST`). `SWE_DISABLE_LANGFUSE=1` to skip.

## Smoke (1 instance)

```bash
export OPENAI_API_KEY='sk-…'   # never commit; env only
export GONGFENG_TOKEN='...'
export SWE_DATASET_PATH=/path/to/swe_bench_pro_fix0319.jsonl
export SWE_INCLUDE_INSTANCE_IDS=astropy__astropy-13236
export SWE_LIMIT=1
export LANGFUSE_PUBLIC_KEY='...'
export LANGFUSE_SECRET_KEY='...'
export LANGFUSE_BASE_URL='...'
# Codex is hard-pinned to 0.154.0; wrap_plaintext_reasoning_content is False in generator.py.

cd /path/to/nexus
uv run nexus run-batch \
  -g docs/examples/configs/swepro-codex-gpt56/generator.py \
  -d ./swepro_codex_gpt56_smoke.db \
  --max-concurrent 1
```

## Full SWE-Pro (731)

Omit `SWE_INCLUDE_INSTANCE_IDS`; set `--max-concurrent` as needed.
