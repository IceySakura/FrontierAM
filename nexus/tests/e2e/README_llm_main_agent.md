# LLM client, Main Agent, and GUI E2E

`test_main_agent_text_thinking.py` calls real model endpoints without creating a runtime or starting a GUI. It covers:

- `VertexClient(protocol=anthropic)` with Claude Opus
- `VertexClient(protocol=openai)` against an OpenAI Chat Completions endpoint
- the official OpenAI Responses API client
- `LiteLLMClient` with OpenAI
- `LiteLLMClient` with Anthropic

For every client, the suite checks a real thinking response, four direct provider rounds, and a four-round Main Agent
tool loop. Non-Responses clients also exercise real streaming. Additional cases cover the deprecated
`vertex_anthropic` alias and a real Responses request with `reasoning.effort=xhigh`.

`chat_tracker` remains trajectory-only for every client. Direct Vertex and Responses tests attach a test-only span
processor to compare each sanitized native provider response with the next native request. Fresh batch tests validate
the production persistence path directly by reading `provider_call` updates from SQLite `run_trace_events`. LiteLLM
remains on its `main` implementation and is validated through output, thinking, tool continuity, and trajectories.

Credentials are read at runtime from an existing Nexus batch SQLite database. The selected run must contain a resolved
`openai_clients.main.config.client_args.api_key` and `base_url`; environment placeholders are rejected. No credential is
copied into the repository or test output.

```bash
NEXUS_LLM_E2E_ACCOUNT_DB=/absolute/path/to/batch.db \
NEXUS_LLM_E2E_ACCOUNT_RUN_ID=run-id-with-resolved-account \
uv run --with 'litellm>=1.86.0,<1.92.0' --with boto3 \
  pytest tests/e2e/test_main_agent_text_thinking.py -m 'e2e and llm' -q
```

Persist a separate real four-round Task, its trajectory-only `chat_tracker` result, and raw trace events into a fresh
SQLite database for every client:

```bash
NEXUS_LLM_E2E_ACCOUNT_DB=/absolute/path/to/account-batch.db \
NEXUS_LLM_E2E_ACCOUNT_RUN_ID=run-id-with-resolved-account \
NEXUS_MAIN_AGENT_BATCH_E2E_OUTPUT=/absolute/output/directory \
uv run --with 'litellm>=1.86.0,<1.92.0' --with boto3 \
  pytest tests/e2e/test_main_agent_batch_persistence.py -m 'e2e and llm' -q
```

`test_claude_code_text_proxy.py` creates a headless Docker runtime and runs the standard `claude_code --print` agent
through Nexus' Anthropic proxy. Its real-provider cases cover text and four sequential Bash rounds with Claude Opus:

```bash
NEXUS_LLM_E2E_ACCOUNT_DB=/absolute/path/to/batch.db \
NEXUS_LLM_E2E_ACCOUNT_RUN_ID=run-id-with-resolved-account \
uv run pytest tests/e2e/test_claude_code_text_proxy.py -m 'e2e and llm' -q
```

The real Docker GUI cases execute four model responses, three actual desktop actions, screenshot feedback, and a final
answer. They write sanitized config, agent output, trajectory-only `chat_tracker` data, test-captured provider calls,
and before/after screenshots:

```bash
NEXUS_RUN_REAL_RESPONSES_GUI_E2E=1 \
NEXUS_RUN_REAL_VERTEX_GUI_E2E=1 \
NEXUS_LLM_E2E_ACCOUNT_DB=/absolute/path/to/batch.db \
NEXUS_LLM_E2E_ACCOUNT_RUN_ID=run-id-with-resolved-account \
uv run pytest \
  tests/e2e/test_main_agent_responses_gui.py \
  tests/e2e/test_main_agent_vertex_gui.py -m 'e2e and llm and gui' -q
```

Optional overrides:

- `NEXUS_VERTEX_ANTHROPIC_E2E_MODEL` (defaults to `claude-opus-4-8`)
- `NEXUS_VERTEX_OPENAI_E2E_MODEL` (defaults to `gpt-5.6-sol`)
- `NEXUS_VERTEX_OPENAI_E2E_BASE_URL` (optional exact SDK base URL; defaults to the account URL plus `/v1`)
- `NEXUS_VERTEX_OPENAI_E2E_GATEWAY_PROVIDER` (defaults to `openai`; set empty for a direct endpoint)
- `NEXUS_VERTEX_OPENAI_E2E_GATEWAY_API` (optional gateway API selector)
- `NEXUS_RESPONSES_E2E_MODEL` (defaults to `gpt-5.6-sol`)
- `NEXUS_LITELLM_OPENAI_E2E_MODEL` (defaults to `gpt-5.6-sol`)
- `NEXUS_MAIN_AGENT_E2E_OUTPUT` (sanitized text/tool trajectory output root)
- `NEXUS_MAIN_AGENT_BATCH_E2E_OUTPUT` (fresh SQLite output root)
- `NEXUS_RESPONSES_XHIGH_E2E_OUTPUT` (sanitized Responses xhigh artifacts)
- `NEXUS_RESPONSES_GUI_E2E_OUTPUT` and `NEXUS_VERTEX_GUI_E2E_OUTPUT` (GUI artifacts)
- `NEXUS_LLM_E2E_TIMEOUT` (defaults to `300` seconds per request)
- `NEXUS_CLAUDE_CODE_E2E_MODEL` (defaults to `claude-opus-4-8`)
- `NEXUS_CLAUDE_CODE_E2E_TIMEOUT` (defaults to `600` seconds)
- `NEXUS_CLAUDE_CODE_E2E_IMAGE` (headless Docker user image)
- `NEXUS_CLAUDE_CODE_E2E_RUNTIME_IMAGE` (defaults to the current Nexus runtime image)
- `NEXUS_CLAUDE_CODE_E2E_VERSION` (defaults to `latest`)

The default SQLite fixture used during development contains a Vertex Anthropic/rawPredict route but not Google OAuth or
a Google Vertex OpenAI-compatible endpoint. Consequently, the default `vertex_openai` case proves the unified client's
OpenAI adapter against the same gateway's Chat Completions path; it does not claim that this fixture validates a Google
Vertex OpenAI upstream. A dedicated Vertex endpoint can be supplied through an appropriate SQLite account record and
the gateway overrides above.

If `NEXUS_LLM_E2E_ACCOUNT_DB` is absent, the live cases are skipped. If it is present but invalid, the test fails with a
credential-safe error.
