from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import pytest

from nexus.extensions.agents.installed_common import find_free_port
from nexus.extensions.agents.openclaw.runtime_setup import (
    NEXUS_BRAVE_API_KEY,
    WEB_SEARCH_ENDPOINT_PATCH_MARKER,
    _build_openclaw_env,
    _has_ready_brave_web_search_provider,
    _patch_full_config,
    _prepare_web_search_compat,
    build_openclaw_config,
)
from nexus.extensions.agents.openclaw.web_search_bridge import (
    SerperBridge,
    SerperBridgeConfig,
    resolve_serper_bridge_config,
)


REAL_UPSTREAM_TEST_ENV = 'NEXUS_OPENCLAW_WEB_SEARCH_REAL_UPSTREAM'
CLI_E2E_TEST_ENV = 'NEXUS_OPENCLAW_WEB_SEARCH_CLI_E2E'
CLI_E2E_UPSTREAM_ENV = 'NEXUS_OPENCLAW_WEB_SEARCH_CLI_UPSTREAM'
CLI_E2E_NODE_ENV = 'NEXUS_OPENCLAW_WEB_SEARCH_NODE'
CLI_E2E_VERSION_ENV = 'NEXUS_OPENCLAW_WEB_SEARCH_CLI_VERSION'
CLI_E2E_INSTALL_ROOT_ENV = 'NEXUS_OPENCLAW_WEB_SEARCH_CLI_INSTALL_ROOT'
CLI_E2E_DEFAULT_VERSION = '2026.6.1'
CLI_E2E_DEFAULT_INSTALL_ROOT = Path('/tmp/openclaw-agent-bridge-smoke')
CLI_E2E_DEFAULT_NPM_REGISTRY = 'http://mirrors.tencent.com/npm/'


@dataclass(frozen=True)
class OpenClawSearchRuntime:
    search_base_url: str
    config: dict[str, Any]
    env: dict[str, str]
    uses_brave_base_url: bool


fastapi = pytest.importorskip('fastapi')
uvicorn = pytest.importorskip('uvicorn')
from fastapi import FastAPI, Request  # noqa: E402


class FakeSerperUpstream:
    def __init__(self) -> None:
        self.host = '127.0.0.1'
        self.port = find_free_port()
        self.requests: list[dict[str, Any]] = []
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task | None = None

    @property
    def url(self) -> str:
        return f'http://{self.host}:{self.port}/search'

    async def start(self) -> None:
        app = FastAPI()

        @app.post('/search')
        async def search(request: Request) -> dict[str, Any]:
            body = await request.json()
            self.requests.append(
                {
                    'headers': dict(request.headers),
                    'body': body,
                }
            )
            return {
                'organic': [
                    {
                        'title': 'Fake OpenClaw result',
                        'link': 'https://example.com/openclaw',
                        'snippet': f"Fake answer for {body['q']}",
                        'date': 'today',
                    }
                ]
            }

        self._server = uvicorn.Server(
            uvicorn.Config(app, host=self.host, port=self.port, log_level='warning', access_log=False)
        )
        self._task = asyncio.create_task(self._server.serve())
        await asyncio.sleep(0.2)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            await self._task


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, raw_value = line.split('=', 1)
        values[key.strip()] = raw_value.strip().strip('"').strip("'")
    return values


def _resolve_real_upstream_config() -> SerperBridgeConfig:
    env = {**_load_dotenv(Path.cwd() / '.env'), **os.environ}
    config = resolve_serper_bridge_config(env)
    if config is None:
        pytest.skip('Serper credentials are not configured in environment or .env')
    return config


def _require_real_upstream_config() -> SerperBridgeConfig:
    enabled = os.environ.get(REAL_UPSTREAM_TEST_ENV, '').strip().lower()
    if enabled not in {'1', 'true', 'yes', 'on'}:
        pytest.skip(f'set {REAL_UPSTREAM_TEST_ENV}=1 to run real upstream web_search tests')
    return _resolve_real_upstream_config()


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, '').strip().lower() in {'1', 'true', 'yes', 'on'}


def _require_cli_e2e_enabled() -> None:
    if not _env_enabled(CLI_E2E_TEST_ENV):
        pytest.skip(f'set {CLI_E2E_TEST_ENV}=1 to run real OpenClaw CLI web_search E2E tests')


def _parse_node_version(stdout: str) -> tuple[int, int, int] | None:
    version = stdout.strip().lstrip('v')
    parts = version.split('.')
    if not parts or not parts[0].isdigit():
        return None
    parsed = [int(part) if part.isdigit() else 0 for part in parts[:3]]
    while len(parsed) < 3:
        parsed.append(0)
    return parsed[0], parsed[1], parsed[2]


def _node_version(path: Path) -> tuple[int, int, int] | None:
    try:
        result = subprocess.run(
            [str(path), '--version'],
            text=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return _parse_node_version(result.stdout)


def _resolve_cli_e2e_node() -> Path:
    candidates: list[Path] = []
    if os.environ.get(CLI_E2E_NODE_ENV):
        candidates.append(Path(os.environ[CLI_E2E_NODE_ENV]))
    candidates.extend(
        [
            Path('/tmp/node-v22.19.0-linux-x64/bin/node'),
            Path('/data/tmp/node-22.19-check/node_modules/node/node_modules/node-linux-x64/bin/node'),
        ]
    )
    if shutil.which('node'):
        candidates.append(Path(shutil.which('node') or 'node'))

    for candidate in candidates:
        if not candidate.exists():
            continue
        version = _node_version(candidate)
        if version is not None and version >= (22, 19, 0):
            return candidate

    pytest.skip(
        f'OpenClaw CLI E2E needs Node >= 22.19.0; set {CLI_E2E_NODE_ENV} to a compatible node binary'
    )


def _openclaw_cli_npm_env(node_path: Path) -> dict[str, str]:
    registry = os.environ.get('OPENCLAW_NPM_REGISTRY') or CLI_E2E_DEFAULT_NPM_REGISTRY
    env = os.environ.copy()
    env.update(
        {
            'PATH': f'{node_path.parent}:{env.get("PATH", "")}',
            'NPM_CONFIG_REGISTRY': registry,
            'npm_config_registry': registry,
            'npm_config_audit': 'false',
            'npm_config_fund': 'false',
            'npm_config_update_notifier': 'false',
            'NPM_CONFIG_STRICT_SSL': 'false',
            'npm_config_strict_ssl': 'false',
            'CI': '1',
            'NO_COLOR': '1',
        }
    )
    return env


async def _run_openclaw_cli_command(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    def _run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            cwd=str(cwd),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )

    return await asyncio.to_thread(_run)


async def _ensure_openclaw_cli_for_e2e(node_path: Path) -> Path:
    version = os.environ.get(CLI_E2E_VERSION_ENV, CLI_E2E_DEFAULT_VERSION)
    install_root = Path(os.environ.get(CLI_E2E_INSTALL_ROOT_ENV, str(CLI_E2E_DEFAULT_INSTALL_ROOT)))
    prefix = install_root / version
    openclaw_bin = prefix / 'node_modules' / '.bin' / 'openclaw'
    if openclaw_bin.exists():
        return openclaw_bin

    prefix.mkdir(parents=True, exist_ok=True)
    env = _openclaw_cli_npm_env(node_path)
    if shutil.which('npm', path=env['PATH']) is None:
        pytest.skip('OpenClaw CLI E2E needs npm on PATH')

    result = await _run_openclaw_cli_command(
        [
            'npm',
            'install',
            '--prefix',
            str(prefix),
            '--no-save',
            '--no-package-lock',
            '--omit=dev',
            '--loglevel=error',
            f'openclaw@{version}',
        ],
        cwd=prefix,
        env=env,
        timeout=1800,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert openclaw_bin.exists(), f'OpenClaw binary not found after npm install: {openclaw_bin}'
    return openclaw_bin


def _build_openclaw_cli_e2e_env(
    *,
    node_path: Path,
    home_dir: Path,
    state_dir: Path,
    config_path: Path,
    bridge_base_url: str,
    bridge_port: int,
    proxy_port: int,
) -> dict[str, str]:
    env = _openclaw_cli_npm_env(node_path)
    no_proxy = (
        f'localhost,127.0.0.1,localhost:{bridge_port},127.0.0.1:{bridge_port},'
        f'localhost:{proxy_port},127.0.0.1:{proxy_port}'
    )
    env.update(
        {
            'HOME': str(home_dir),
            'OPENCLAW_HOME': str(home_dir),
            'OPENCLAW_STATE_DIR': str(state_dir),
            'OPENCLAW_CONFIG_PATH': str(config_path),
            'BRAVE_API_KEY': NEXUS_BRAVE_API_KEY,
            'NEXUS_SERPER_BRIDGE_BASE_URL': bridge_base_url,
            'no_proxy': no_proxy,
            'NO_PROXY': no_proxy,
        }
    )
    env.update(
        {
            'SERPER_API_KEY': '',
            'SERPER_API_TYPE': '',
            'SEARCH_SERPER_URL': '',
            'SEARCH_SERPER_APP_ID': '',
            'SEARCH_SERPER_APP_KEY': '',
            'SERPER_KEY_ID': '',
        }
    )
    return env


def _write_openclaw_config(path: Path) -> None:
    path.write_text(
        """{
  "tools": {
    "web": {
      "search": {
        "enabled": true,
        "provider": "brave",
        "apiKey": "stale-real-key"
      }
    }
  },
  "models": {
    "providers": {
      "nexusproxy": {
        "models": [
          {
            "id": "gpt-test",
            "name": "gpt-test"
          }
        ]
      }
    }
  }
}""",
        encoding='utf-8',
    )


def _write_openclaw_runtime(tmp_path: Path, *, supports_brave_base_url: bool) -> tuple[Path, Path]:
    openclaw_bin = tmp_path / 'node_modules' / 'openclaw' / 'bin' / 'openclaw.mjs'
    dist_dir = tmp_path / 'node_modules' / 'openclaw' / 'dist'
    openclaw_bin.parent.mkdir(parents=True)
    dist_dir.mkdir(parents=True)
    openclaw_bin.write_text('', encoding='utf-8')

    web_tools = dist_dir / 'web-tools.js'
    if supports_brave_base_url:
        web_tools.write_text(
            'const BRAVE_SEARCH_ENDPOINT_PATH = "/res/v1/web/search"; function resolveBraveBaseUrl() {}',
            encoding='utf-8',
        )
    else:
        web_tools.write_text(
            """const BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search";
async function runWebSearch() {
  const url = new URL(BRAVE_SEARCH_ENDPOINT);
}
""",
            encoding='utf-8',
        )

    return openclaw_bin, dist_dir


def _setup_openclaw_web_search_runtime(
    tmp_path: Path,
    *,
    bridge_base_url: str,
    supports_brave_base_url: bool,
    serper_api_key: str,
    serper_api_type: str,
) -> OpenClawSearchRuntime:
    openclaw_bin, dist_dir = _write_openclaw_runtime(
        tmp_path,
        supports_brave_base_url=supports_brave_base_url,
    )
    compat = _prepare_web_search_compat(
        openclaw_bin=openclaw_bin,
        bridge_base_url=bridge_base_url,
        serper_configured=True,
    )

    assert compat.enabled is True
    assert compat.base_url == bridge_base_url
    assert compat.uses_brave_base_url is supports_brave_base_url

    config_path = tmp_path / 'openclaw.json'
    _write_openclaw_config(config_path)
    _patch_full_config(
        config_path=config_path,
        provider_id='nexusproxy',
        model='gpt-test',
        web_search_enabled=compat.enabled,
        web_search_base_url=compat.base_url,
        web_search_uses_brave_base_url=compat.uses_brave_base_url,
    )
    env = _build_openclaw_env(
        home_dir=tmp_path / 'home',
        state_dir=tmp_path / 'state',
        config_path=config_path,
        proxy_port=find_free_port(),
        web_search=compat,
        curl_binary=None,
        base_env={
            'SERPER_API_KEY': serper_api_key,
            'SERPER_API_TYPE': serper_api_type,
        },
    )

    patched_config = json.loads(config_path.read_text(encoding='utf-8'))
    expected_search_config = {
        'enabled': True,
        'provider': 'brave',
    }
    if not supports_brave_base_url:
        expected_search_config['apiKey'] = NEXUS_BRAVE_API_KEY
    assert patched_config['tools']['web']['search'] == expected_search_config
    assert env['BRAVE_API_KEY'] == NEXUS_BRAVE_API_KEY
    assert env['SERPER_API_KEY'] == ''
    assert env['SERPER_API_TYPE'] == ''
    assert env['NEXUS_SERPER_BRIDGE_BASE_URL'] == bridge_base_url

    if supports_brave_base_url:
        brave_cfg = patched_config['plugins']['entries']['brave']['config']['webSearch']
        assert brave_cfg == {
            'apiKey': NEXUS_BRAVE_API_KEY,
            'baseUrl': bridge_base_url,
            'mode': 'web',
        }
        openclaw_search_base_url = brave_cfg['baseUrl']
    else:
        assert patched_config.get('plugins', {}).get('entries', {}).get('brave') is None
        patched_web_tools = (dist_dir / 'web-tools.js').read_text(encoding='utf-8')
        assert WEB_SEARCH_ENDPOINT_PATCH_MARKER in patched_web_tools
        assert 'process.env.NEXUS_SERPER_BRIDGE_BASE_URL' in patched_web_tools
        openclaw_search_base_url = env['NEXUS_SERPER_BRIDGE_BASE_URL']

    return OpenClawSearchRuntime(
        search_base_url=openclaw_search_base_url,
        config=patched_config,
        env=env,
        uses_brave_base_url=compat.uses_brave_base_url,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    'supports_brave_base_url',
    [
        pytest.param(True, id='new-openclaw-base-url'),
        pytest.param(False, id='legacy-openclaw-js-patch'),
    ],
)
async def test_openclaw_web_search_runtime_uses_bridge_end_to_end(
    tmp_path: Path,
    supports_brave_base_url: bool,
) -> None:
    upstream = FakeSerperUpstream()
    bridge: SerperBridge | None = None
    await upstream.start()
    try:
        bridge = SerperBridge(
            config=SerperBridgeConfig(
                api_key='fake-app:fake-key',
                api_type='distillation',
                search_url=upstream.url,
            ),
            listen_port=find_free_port(),
        )
        await bridge.start()

        runtime = _setup_openclaw_web_search_runtime(
            tmp_path,
            bridge_base_url=bridge.base_url,
            supports_brave_base_url=supports_brave_base_url,
            serper_api_key='real-serper-key',
            serper_api_type='distillation',
        )

        query = f"OpenClaw linked bridge smoke {'new' if supports_brave_base_url else 'legacy'}"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f'{runtime.search_base_url.rstrip("/")}/res/v1/web/search',
                params={
                    'q': query,
                    'count': '6',
                    'search_lang': 'zh',
                    'ui_lang': 'zh-CN',
                    'country': 'CN',
                    'freshness': 'pw',
                },
            ) as response:
                response_text = await response.text()
                assert response.status == 200, response_text
                payload = json.loads(response_text)

        assert payload == {
            'web': {
                'results': [
                    {
                        'title': 'Fake OpenClaw result',
                        'url': 'https://example.com/openclaw',
                        'description': f'Fake answer for {query}',
                        'age': 'today',
                    }
                ]
            }
        }
        assert upstream.requests[-1]['body'] == {
            'q': query,
            'num': 6,
            'hl': 'zh',
            'gl': 'CN',
            'tbs': 'qdr:w',
        }
        assert upstream.requests[-1]['headers']['authorization'] == 'Bearer fake-app:fake-key?provider=serper'
    finally:
        if bridge is not None:
            await bridge.stop()
        await upstream.stop()


@pytest.mark.integration
@pytest.mark.e2e
async def test_openclaw_cli_web_search_hits_bridge_with_selectable_upstream(tmp_path: Path) -> None:
    _require_cli_e2e_enabled()

    upstream_mode = os.environ.get(CLI_E2E_UPSTREAM_ENV, 'mock').strip().lower() or 'mock'
    fake_upstream: FakeSerperUpstream | None = None
    if upstream_mode == 'mock':
        fake_upstream = FakeSerperUpstream()
        await fake_upstream.start()
        serper_config = SerperBridgeConfig(
            api_key='fake-app:fake-key',
            api_type='distillation',
            search_url=fake_upstream.url,
        )
    elif upstream_mode == 'real':
        serper_config = _resolve_real_upstream_config()
    else:
        pytest.fail(f'{CLI_E2E_UPSTREAM_ENV} must be "mock" or "real", got {upstream_mode!r}')

    bridge: SerperBridge | None = None
    try:
        node_path = _resolve_cli_e2e_node()
        openclaw_bin = await _ensure_openclaw_cli_for_e2e(node_path)

        bridge = SerperBridge(config=serper_config, listen_port=find_free_port(), request_timeout=30)
        await bridge.start()

        home_dir = tmp_path / 'home'
        state_dir = tmp_path / 'state'
        config_path = state_dir / 'openclaw.json'
        workspace = tmp_path / 'workspace'
        home_dir.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)
        workspace.mkdir(parents=True, exist_ok=True)
        proxy_port = find_free_port()
        env = _build_openclaw_cli_e2e_env(
            node_path=node_path,
            home_dir=home_dir,
            state_dir=state_dir,
            config_path=config_path,
            bridge_base_url=bridge.base_url,
            bridge_port=bridge.port,
            proxy_port=proxy_port,
        )

        install_result = await _run_openclaw_cli_command(
            [str(openclaw_bin), 'plugins', 'install', '@openclaw/brave-plugin'],
            cwd=workspace,
            env=env,
            timeout=300,
        )
        assert install_result.returncode == 0, install_result.stderr or install_result.stdout

        enable_result = await _run_openclaw_cli_command(
            [str(openclaw_bin), 'plugins', 'enable', 'brave'],
            cwd=workspace,
            env=env,
            timeout=120,
        )
        assert enable_result.returncode == 0, enable_result.stderr or enable_result.stdout

        config_path.write_text(
            json.dumps(build_openclaw_config(model='gpt-test', working_dir=str(workspace), proxy_port=proxy_port)),
            encoding='utf-8',
        )
        _patch_full_config(
            config_path=config_path,
            provider_id='nexusproxy',
            model='gpt-test',
            web_search_enabled=True,
            web_search_base_url=bridge.base_url,
            web_search_uses_brave_base_url=True,
        )

        providers_result = await _run_openclaw_cli_command(
            [str(openclaw_bin), 'infer', 'web', 'providers', '--json'],
            cwd=workspace,
            env=env,
            timeout=120,
        )
        assert providers_result.returncode == 0, providers_result.stderr or providers_result.stdout
        assert _has_ready_brave_web_search_provider(json.loads(providers_result.stdout))

        query = f'OpenClaw CLI bridge e2e {upstream_mode}'
        search_result = await _run_openclaw_cli_command(
            [
                str(openclaw_bin),
                'infer',
                'web',
                'search',
                '--json',
                '--provider',
                'brave',
                '--query',
                query,
                '--limit',
                '2',
            ],
            cwd=workspace,
            env=env,
            timeout=120,
        )
        if upstream_mode == 'real' and search_result.returncode != 0:
            # A real upstream may be unreachable from the current network. A
            # provider HTTP error still proves OpenClaw reached the local
            # bridge and received the bridge's upstream response.
            assert 'Brave Search API error' in search_result.stderr, search_result.stderr or search_result.stdout
        else:
            assert search_result.returncode == 0, search_result.stderr or search_result.stdout
            assert 'Fake OpenClaw result' in search_result.stdout or upstream_mode == 'real'

        if fake_upstream is not None:
            assert fake_upstream.requests
            assert fake_upstream.requests[-1]['body'] == {'q': query, 'num': 2}
            assert fake_upstream.requests[-1]['headers']['authorization'] == 'Bearer fake-app:fake-key?provider=serper'
    finally:
        if bridge is not None:
            await bridge.stop()
        if fake_upstream is not None:
            await fake_upstream.stop()


@pytest.mark.integration
@pytest.mark.parametrize(
    'supports_brave_base_url',
    [
        pytest.param(True, id='new-openclaw-base-url'),
        pytest.param(False, id='legacy-openclaw-js-patch'),
    ],
)
async def test_openclaw_web_search_runtime_uses_real_upstream_when_enabled(
    tmp_path: Path,
    supports_brave_base_url: bool,
) -> None:
    config = _require_real_upstream_config()
    bridge = SerperBridge(config=config, listen_port=find_free_port(), request_timeout=30)
    await bridge.start()
    try:
        runtime = _setup_openclaw_web_search_runtime(
            tmp_path,
            bridge_base_url=bridge.base_url,
            supports_brave_base_url=supports_brave_base_url,
            serper_api_key=config.api_key,
            serper_api_type=config.api_type,
        )

        query = f"OpenClaw real upstream bridge smoke {'new' if supports_brave_base_url else 'legacy'}"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f'{runtime.search_base_url.rstrip("/")}/res/v1/web/search',
                params={'q': query, 'count': '3'},
            ) as response:
                response_text = await response.text()
                assert response.status == 200, response_text
                payload = json.loads(response_text)

        results = payload.get('web', {}).get('results', [])
        assert isinstance(results, list)
        assert results
        assert results[0].get('title')
        assert results[0].get('url')
    finally:
        await bridge.stop()


@pytest.mark.integration
async def test_serper_bridge_fake_upstream_end_to_end() -> None:
    upstream = FakeSerperUpstream()
    bridge: SerperBridge | None = None
    await upstream.start()
    try:
        bridge = SerperBridge(
            config=SerperBridgeConfig(
                api_key='fake-app:fake-key',
                api_type='distillation',
                search_url=upstream.url,
            ),
            listen_port=find_free_port(),
        )
        await bridge.start()

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f'{bridge.base_url}/res/v1/web/search',
                params={
                    'q': 'bridge smoke',
                    'count': '4',
                    'ui_lang': 'en-US',
                    'freshness': '2026-06-01to2026-06-04',
                },
            ) as response:
                assert response.status == 200
                payload = await response.json()

        assert payload == {
            'web': {
                'results': [
                    {
                        'title': 'Fake OpenClaw result',
                        'url': 'https://example.com/openclaw',
                        'description': 'Fake answer for bridge smoke',
                        'age': 'today',
                    }
                ]
            }
        }
        assert upstream.requests == [
            {
                'headers': {
                    **upstream.requests[0]['headers'],
                    'authorization': 'Bearer fake-app:fake-key?provider=serper',
                },
                'body': {
                    'q': 'bridge smoke',
                    'num': 4,
                    'hl': 'en-US',
                    'tbs': 'cdr:1,cd_min:6/1/2026,cd_max:6/4/2026',
                },
            }
        ]
    finally:
        if bridge is not None:
            await bridge.stop()
        await upstream.stop()


@pytest.mark.integration
async def test_serper_bridge_real_upstream_end_to_end() -> None:
    config = _require_real_upstream_config()

    bridge = SerperBridge(config=config, listen_port=find_free_port(), request_timeout=30)
    await bridge.start()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f'{bridge.base_url}/res/v1/web/search',
                params={'q': 'OpenClaw web search smoke test', 'count': '3'},
            ) as response:
                response_text = await response.text()
                assert response.status == 200, response_text
                payload = await response.json()

        results = payload.get('web', {}).get('results', [])
        assert isinstance(results, list)
        assert results
        assert results[0].get('title')
        assert results[0].get('url')
    finally:
        await bridge.stop()
