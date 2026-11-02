# OpenClaw Web Search Serper Bridge Design

Status: Implemented
Date: 2026-06-04

## Goal

OpenClaw's `web_search` tool is Brave-compatible. Nexus should support Serper
and the internal Serper-compatible gateway without injecting Serper-specific
logic into the OpenClaw JavaScript bundle.

The desired runtime path is:

```text
OpenClaw web_search
  -> http://127.0.0.1:<bridge_port>/res/v1/web/search
  -> Serper public API or internal Serper-compatible gateway
```

The local bridge exposes a Brave Search API subset and converts requests to a
Serper-compatible POST upstream.

## Components

### SerperBridge

Implementation:

- `nexus/extensions/agents/openclaw/web_search_bridge.py`

Endpoints:

- `GET /health`
- `GET /res/v1/web/search?q=<query>&count=<n>`

Supported Brave-style query parameters:

- `q` -> Serper `q`
- `count` -> Serper `num`, clamped to `1..10`
- `search_lang`, `language`, `ui_lang` -> Serper `hl` when they can be
  represented as one compatible locale
- `country` -> Serper `gl`
- `freshness=pd|pw|pm|py` -> Serper `tbs=qdr:d|qdr:w|qdr:m|qdr:y`
- `freshness=YYYY-MM-DDtoYYYY-MM-DD` -> Serper custom date-range `tbs`

Unsupported or conflicting Brave query parameters return HTTP 400 instead of
being silently dropped.

Bridge response shape:

```json
{
  "web": {
    "results": [
      {
        "title": "Example",
        "url": "https://example.com",
        "description": "Snippet",
        "age": "today"
      }
    ]
  }
}
```

Serper public mode:

- `SERPER_API_TYPE` unset or not `distillation`
- upstream URL: `https://google.serper.dev/search` unless `SEARCH_SERPER_URL`
  overrides it
- auth header: `X-API-KEY: <SERPER_API_KEY>`

Internal gateway mode:

- `SERPER_API_TYPE=distillation`
- upstream URL: required `SEARCH_SERPER_URL`; Nexus does not ship an internal
  gateway default
- auth header: `Authorization: Bearer <APP_ID>:<APP_KEY>?provider=serper`
- if `SERPER_API_KEY` is unset, the bridge derives it from
  `SEARCH_SERPER_APP_ID` or `SERPER_KEY_ID`, plus `SEARCH_SERPER_APP_KEY`

The bridge sanitizes upstream error details before returning them to OpenClaw.

### Runtime Setup

Implementation:

- `nexus/extensions/agents/openclaw/runtime_setup.py`

Runtime setup starts the bridge only when Serper credentials are present. If the
bridge cannot start, or OpenClaw compatibility setup fails, Nexus disables
`tools.web.search` instead of starting OpenClaw with a broken search config.
After writing the OpenClaw config, Nexus also runs
`openclaw infer web providers --json` and keeps web search enabled only when
the `brave` provider is reported as available and configured. This catches the
case where OpenClaw accepts the config shape but the Brave plugin is not
installed, which otherwise surfaces inside the agent as a generic
`web_search` tool error.
For new plugin-schema OpenClaw builds, Nexus first attempts to install and
enable `@openclaw/brave-plugin` using the configured npm registry. The default
registry is the Tencent mirror `http://mirrors.tencent.com/npm/`; if plugin
installation or provider preflight still fails, Nexus disables `web_search`
for that run.
Recent OpenClaw builds require Node.js 22.19 or newer; Nexus fails early during
OpenClaw installation if the runtime Node.js is older.

OpenClaw child process env:

- `NEXUS_SERPER_BRIDGE_BASE_URL=http://127.0.0.1:<bridge_port>` when enabled
- `BRAVE_API_KEY=nexus-serper-bridge`
- all Serper bridge env vars are explicitly blanked so real Serper credentials
  and gateway settings are not exposed to OpenClaw
- `no_proxy` and `NO_PROXY` include both the local OpenAI proxy port and the
  local bridge port

### OpenClaw Version Compatibility

New OpenClaw builds that expose Brave provider `baseUrl` support:

- no JavaScript patch
- Nexus attempts `openclaw plugins install @openclaw/brave-plugin` and
  `openclaw plugins enable brave` before writing the Brave web search config
- plugin install uses the configured npm registry, defaulting to
  `http://mirrors.tencent.com/npm/`
- if the `brave` plugin is still missing or not configured, provider preflight
  fails and Nexus disables `web_search` for that run instead of exposing a tool
  that will always return `fetch failed`
- config writes:

```json
{
  "tools": {
    "web": {
      "search": {
        "enabled": true,
        "provider": "brave"
      }
    }
  },
  "plugins": {
    "entries": {
      "brave": {
        "enabled": true,
        "config": {
          "webSearch": {
            "apiKey": "nexus-serper-bridge",
            "baseUrl": "http://127.0.0.1:<bridge_port>",
            "mode": "web"
          }
        }
      }
    }
  }
}
```

`mode = "web"` is required so OpenClaw uses `/res/v1/web/search` rather than
the Brave LLM-context endpoint.

Legacy OpenClaw builds:

- do not receive `plugins.entries.brave.config.webSearch.baseUrl`
- Nexus patches only the `BRAVE_SEARCH_ENDPOINT` constant to read:

```js
process.env.NEXUS_SERPER_BRIDGE_BASE_URL + "/res/v1/web/search"
```

The legacy patch is idempotent and accepts already env-based endpoint patches.
If an old direct Serper JavaScript patch is detected, Nexus disables
`web_search` rather than stacking incompatible patches.

## WildClaw Integration

Implementation:

- `nexus/extensions/tasks/wildclaw_bench/generator.py`

The WildClaw generator passes these search env vars through to the runtime
container and Docker `extra_args`:

- `SERPER_API_KEY`
- `SERPER_API_TYPE`
- `SEARCH_SERPER_URL`
- `SEARCH_SERPER_APP_ID`
- `SEARCH_SERPER_APP_KEY`
- `SERPER_KEY_ID`

The runtime container still clears generic proxy env vars for the agent process
to avoid hijacking OpenClaw's local LLM proxy traffic. The bridge itself uses
the process environment for upstream HTTP proxy discovery when such proxy env
vars are available.

## Local `.env` Template

Do not commit real credentials. Put local values in an ignored `.env` file at
the repository root or worktree root.

Internal gateway example:

```bash
SERPER_API_TYPE="distillation"
SEARCH_SERPER_URL="http://your-serper-gateway/search"
SEARCH_SERPER_APP_ID="<app-id>"
SEARCH_SERPER_APP_KEY="<app-key>"
SERPER_KEY_ID="<app-id>"
SERPER_API_KEY="<app-id>:<app-key>"
```

Public Serper example:

```bash
SERPER_API_KEY="<serper-api-key>"
```

## Validation

### Focused Tests

Run the bridge/runtime/generator tests plus fake upstream smoke:

```bash
uv run --extra dev --extra runtime pytest \
  tests/unit/extensions/agents/test_openclaw_installer.py \
  tests/unit/extensions/agents/test_openclaw_web_search_bridge.py \
  tests/unit/extensions/agents/test_openclaw_runtime_setup.py \
  tests/unit/extensions/tasks/test_wildclaw_generator_web_search_env.py \
  tests/integration/openclaw/test_web_search_bridge_smoke.py::test_serper_bridge_fake_upstream_end_to_end \
  tests/integration/openclaw/test_web_search_bridge_smoke.py::test_openclaw_web_search_runtime_uses_bridge_end_to_end \
  -q
```

Expected result:

```text
39 passed, 2 warnings
```

The warnings are deprecation warnings from `websockets` / `uvicorn` and do not
affect bridge behavior.

The fake smoke starts two local HTTP services:

1. a fake Serper upstream
2. the Nexus SerperBridge

It verifies the complete request path:

```text
Brave-style GET -> SerperBridge -> Serper-compatible POST -> Brave-style JSON
```

### OpenClaw CLI E2E Smoke

Run the full OpenClaw CLI path only when you want to exercise a real OpenClaw
package and plugin install:

```bash
NEXUS_OPENCLAW_WEB_SEARCH_CLI_E2E=1 \
uv run --extra dev pytest \
  tests/integration/openclaw/test_web_search_bridge_smoke.py::test_openclaw_cli_web_search_hits_bridge_with_selectable_upstream \
  -q
```

Default behavior uses a fake Serper upstream:

```text
OpenClaw CLI -> SerperBridge -> fake Serper upstream
```

Set `NEXUS_OPENCLAW_WEB_SEARCH_CLI_UPSTREAM=real` to make the bridge exit to
the real upstream described by `.env` / shell environment variables instead:

```bash
NEXUS_OPENCLAW_WEB_SEARCH_CLI_E2E=1 \
NEXUS_OPENCLAW_WEB_SEARCH_CLI_UPSTREAM=real \
uv run --extra dev pytest \
  tests/integration/openclaw/test_web_search_bridge_smoke.py::test_openclaw_cli_web_search_hits_bridge_with_selectable_upstream \
  -q
```

Useful overrides:

- `NEXUS_OPENCLAW_WEB_SEARCH_NODE=/path/to/node`: Node.js 22.19+ binary
- `NEXUS_OPENCLAW_WEB_SEARCH_CLI_VERSION=2026.6.1`: OpenClaw npm version
- `NEXUS_OPENCLAW_WEB_SEARCH_CLI_INSTALL_ROOT=/tmp/openclaw-agent-bridge-smoke`
- `OPENCLAW_NPM_REGISTRY=http://mirrors.tencent.com/npm/`

The test installs/enables `@openclaw/brave-plugin`, verifies
`openclaw infer web providers --json`, then runs
`openclaw infer web search --provider brave`. In mock mode it also asserts that
the fake Serper upstream received the translated Serper POST.
In real mode, a successful search is accepted; if the upstream is unreachable,
the test also accepts an OpenClaw `Brave Search API error` because that error is
the bridge's upstream response and proves the OpenClaw request reached the
bridge.

### Real Upstream Smoke

Run this only from a network environment that can reach the selected upstream:

```bash
NEXUS_OPENCLAW_WEB_SEARCH_REAL_UPSTREAM=1 \
uv run --extra dev --extra runtime pytest \
  tests/integration/openclaw/test_web_search_bridge_smoke.py::test_serper_bridge_real_upstream_end_to_end \
  -q
```

Expected result in a network-enabled environment:

```text
1 passed
```

The test reads `.env`, starts the bridge, calls
`/res/v1/web/search?q=OpenClaw web search smoke test&count=3`, and asserts that
the bridge returns at least one result with `title` and `url`.

Shell environment variables override values from `.env`, so temporary command
line overrides take precedence during local smoke runs.

If the internal gateway is unreachable from the current host, the failure may
look like:

```text
HTTP 403
DevCloud机器访问IDC环境需要通过igate开通访问权限
```

That indicates a DevCloud-to-IDC network permission issue. It does not prove a
SerperBridge request-mapping failure. Re-run the smoke from a host or runtime
with the required igate access.

### OpenClaw Package Smoke

Use real npm OpenClaw packages to confirm version compatibility. The command
downloads Node.js 22.19 when needed because recent OpenClaw packages require
that runtime:

```bash
set -euo pipefail

NODE_DIR=/tmp/node-v22.19.0-linux-x64
if [ ! -x "$NODE_DIR/bin/node" ]; then
  curl -fsSL https://nodejs.org/dist/v22.19.0/node-v22.19.0-linux-x64.tar.xz \
    -o /tmp/node-v22.19.0-linux-x64.tar.xz
  tar -C /tmp -xf /tmp/node-v22.19.0-linux-x64.tar.xz
fi
export PATH="$NODE_DIR/bin:$PATH"

SMOKE_ROOT=/tmp/openclaw-real-package-smoke
mkdir -p "$SMOKE_ROOT"
for version in 2026.3.11 2026.5.12 2026.5.28 2026.6.1 2026.6.2-beta.1; do
  prefix="$SMOKE_ROOT/$version"
  if [ ! -x "$prefix/node_modules/.bin/openclaw" ]; then
    rm -rf "$prefix"
    mkdir -p "$prefix"
    npm install --prefix "$prefix" "openclaw@$version" --no-audit --no-fund
  fi
  "$prefix/node_modules/.bin/openclaw" --version
done

uv run python - <<'PY'
import json
import subprocess
from pathlib import Path

from nexus.extensions.agents.openclaw.runtime_setup import (
    NEXUS_BRAVE_API_KEY,
    WEB_SEARCH_ENDPOINT_PATCH_MARKER,
    _find_openclaw_dist_dir,
    _get_js_candidates,
    _patch_full_config,
    _prepare_web_search_compat,
)

versions = ['2026.3.11', '2026.5.12', '2026.5.28', '2026.6.1', '2026.6.2-beta.1']
root = Path('/tmp/openclaw-real-package-smoke')
node = '/tmp/node-v22.19.0-linux-x64/bin/node'

for version in versions:
    openclaw_bin = root / version / 'node_modules' / '.bin' / 'openclaw'
    dist_dir = _find_openclaw_dist_dir(openclaw_bin)
    assert dist_dir is not None, f'{version}: dist not found'
    compat = _prepare_web_search_compat(
        openclaw_bin=openclaw_bin,
        bridge_base_url='http://127.0.0.1:45678',
        serper_configured=True,
    )
    marker_files = [
        Path(path)
        for path in _get_js_candidates(dist_dir)
        if WEB_SEARCH_ENDPOINT_PATCH_MARKER in Path(path).read_text(encoding='utf-8', errors='ignore')
    ]

    cfg_path = root / version / 'openclaw-smoke-config.json'
    cfg_path.write_text(
        json.dumps({
            'tools': {'web': {'search': {'enabled': True, 'provider': 'brave', 'apiKey': 'stale'}}},
            'models': {'providers': {'nexusproxy': {'models': [{'id': 'gpt-test'}]}}},
        }),
        encoding='utf-8',
    )
    _patch_full_config(
        config_path=cfg_path,
        provider_id='nexusproxy',
        model='gpt-test',
        web_search_enabled=compat.enabled,
        web_search_base_url=compat.base_url,
        web_search_uses_brave_base_url=compat.uses_brave_base_url,
    )
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    search_cfg = cfg['tools']['web']['search']
    plugin = cfg.get('plugins', {}).get('entries', {}).get('brave')

    if compat.uses_brave_base_url:
        assert search_cfg == {'enabled': True, 'provider': 'brave'}, (version, search_cfg)
        assert plugin['config']['webSearch'] == {
            'apiKey': NEXUS_BRAVE_API_KEY,
            'baseUrl': 'http://127.0.0.1:45678',
            'mode': 'web',
        }, (version, plugin)
        assert not marker_files, (version, [p.name for p in marker_files])
    else:
        assert search_cfg == {
            'enabled': True,
            'provider': 'brave',
            'apiKey': NEXUS_BRAVE_API_KEY,
        }, (version, search_cfg)
        assert plugin is None, (version, plugin)
        assert marker_files, version
        for path in marker_files:
            subprocess.run([node, '--check', str(path)], check=True)

    print(version, compat, [p.name for p in marker_files])
PY
```

Expected behavior:

- `2026.3.11`: legacy endpoint patch is applied; patched JS passes
  `node --check`.
- `2026.5.12` and newer tested versions: no JS patch is applied; config uses
  `plugins.entries.brave.config.webSearch`, sets `mode = "web"`, and does not
  write legacy `tools.web.search.apiKey`.

### Lint

```bash
uv run --extra dev ruff check \
  nexus/extensions/agents/openclaw/runtime_setup.py \
  nexus/extensions/agents/openclaw/web_search_bridge.py \
  nexus/extensions/tasks/wildclaw_bench/generator.py \
  tests/unit/extensions/agents/test_openclaw_runtime_setup.py \
  tests/unit/extensions/agents/test_openclaw_web_search_bridge.py \
  tests/unit/extensions/tasks/test_wildclaw_generator_web_search_env.py \
  tests/integration/openclaw/test_web_search_bridge_smoke.py
```

Expected result:

```text
All checks passed!
```

## Troubleshooting

- `web_search` disabled in config: check that Serper credentials are present,
  the bridge can start, and `openclaw infer web providers --json` reports
  `brave` as available and configured.
- `brave` missing from provider preflight: on new OpenClaw builds, check the
  preceding `openclaw plugins install @openclaw/brave-plugin` and
  `openclaw plugins enable brave` commands, the configured npm registry, and
  whether the runtime can reach that registry.
- OpenClaw config invalid on old versions: verify old versions do not receive
  `plugins.entries.brave.config.webSearch.baseUrl`.
- New OpenClaw calls `/res/v1/llm/context`: verify
  `plugins.entries.brave.config.webSearch.mode` is `web`.
- OpenClaw sees real Serper credentials or gateway settings: verify child env
  blanks the Serper bridge vars and uses only
  `BRAVE_API_KEY=nexus-serper-bridge`.
- Internal gateway returns IDC 403: run from a host/runtime with igate access or
  configure the required network proxy before starting the bridge.
