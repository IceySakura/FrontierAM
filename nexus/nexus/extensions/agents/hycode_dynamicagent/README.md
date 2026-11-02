# HyCode DynamicAgent in Nexus

This extension runs `@tencent/hycode-dynamicagent` the same way the `hycode`
extension runs `@tencent/hycode`: Nexus installs the npm package inside the
runtime sandbox, starts the Anthropic proxy, then launches the package CLI.

It does not upload a host-side OpenCC binary or copy a host-side plugin folder.

## Single Run

Use `agent_name: hycode_dynamicagent` in a Nexus config. The important CLI args
are the selected direct agent and the native web-tool guard:

```yaml
agent_name: hycode_dynamicagent
agent_config:
  working_dir: /testbed
  version: latest
  npm_registry: https://mirrors.tencent.com/npm/
  user_prompt: Create a 3D rendered Transformer architecture diagram.
  extra_env:
    CODE_ARENA_ROOT: /testbed
  extra_cli_args:
    - --print
    - --input-format
    - text
    - --output-format
    - stream-json
    - --verbose
    - --no-session-persistence
    - --permission-mode
    - bypassPermissions
    - --disallowedTools
    - WebSearch
    - WebFetch
    - --agent
    - code-arena-mcp
```

`WebSearch` and `WebFetch` here are Claude Code native tools. The Code Arena
MCP tools `web_search` and `web_fetch` remain available when the selected
harness exposes them.

## Batch Generator

The generator is:

```bash
./nexus/extensions/agents/hycode_dynamicagent/generator.py
```

It can fan out across bundled Code Arena harnesses:

- `code-arena-mcp`
- `code-arena-html`
- `code-arena-react`
- `code-arena-nextjs-fullstack`

Set the Langfuse credentials in the environment before running. Keep the secret
out of committed files and batch generator source.

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_MODEL="deepseek-chat"

export LANGFUSE_SECRET_KEY="..."
export LANGFUSE_PUBLIC_KEY="..."
export LANGFUSE_BASE_URL="http://21.91.27.43:3000"

export HYCODE_DYNAMICAGENT_PROMPT="Create a 3D rendered Transformer model architecture diagram."
export HYCODE_DYNAMICAGENT_HARNESSES="code-arena-mcp,code-arena-html,code-arena-react,code-arena-nextjs-fullstack"
export HYCODE_DYNAMICAGENT_EXPERIMENT="hycode_dynamicagent_code_arena_transformer"

uv run nexus run-batch \
  -g ./nexus/extensions/agents/hycode_dynamicagent/generator.py \
  -d ./outputs/hycode_dynamicagent_code_arena_transformer.db \
  --max-concurrent 1
```

Useful environment variables:

- `OPENAI_API_KEY`: required model API key used by the OpenAI-compatible client.
- `OPENAI_MODEL`: OpenAI client model, default `deepseek-chat`.
- `OPENAI_BASE_URL`: OpenAI client base URL, default `https://api.deepseek.com`.
- `OPENAI_CLIENT_NAME`: Nexus OpenAI client implementation, default `openai`.
- `OPENAI_MAX_TOKENS`: max tokens for each model request, default `4000`.
- `OPENAI_MAX_RETRIES`: client retry count, default `3`.
- `OPENAI_REQUEST_ARGS_JSON`: JSON object merged into request args.
- `HYCODE_DYNAMICAGENT_HARNESSES`: comma-separated harness names or `all`.
- `HYCODE_DYNAMICAGENT_LIMIT`: cap generated cases for a smoke run.
- `HYCODE_DYNAMICAGENT_REPEAT`: repeat each harness N times.
- `HYCODE_DYNAMICAGENT_PACKAGE_VERSION`: npm package version, default `latest`.
- `HYCODE_DYNAMICAGENT_NPM_REGISTRY`: npm registry, default `https://mirrors.tencent.com/npm/`.
- `HYCODE_DYNAMICAGENT_BOOTSTRAP_TEMPLATES`: default `true`; creates minimal React/Next starter files before those harnesses run.
- `HYCODE_DYNAMICAGENT_DISABLE_LANGFUSE`: set `1` to run without Langfuse.

For a fast smoke:

```bash
HYCODE_DYNAMICAGENT_HARNESSES=code-arena-mcp \
HYCODE_DYNAMICAGENT_LIMIT=1 \
uv run nexus run-batch \
  -g ./nexus/extensions/agents/hycode_dynamicagent/generator.py \
  -d /tmp/hycode_dynamicagent_smoke.db \
  --max-concurrent 1
```
