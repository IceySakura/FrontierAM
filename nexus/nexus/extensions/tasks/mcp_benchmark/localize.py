from __future__ import annotations

import copy

from typing import Any

from .constants import (
    ATLAS_IMAGE_ROOT,
    BENCHMARK_IMAGE_HELPERS,
    BENCHMARK_IMAGE_NPM_BIN,
    BENCHMARK_IMAGE_NPM_ROOT,
    BENCHMARK_IMAGE_PY311,
    BENCHMARK_IMAGE_PY311_BIN,
    BENCHMARK_IMAGE_PY312_BIN,
    BENCHMARK_IMAGE_PY313_BIN,
    BENCHMARK_IMAGE_VENV_BIN,
    BENCHMARK_LOCAL_SERVERS,
    TOOLATHLON_IMAGE_ROOT,
    UNIVERSE_IMAGE_ROOT,
)
from .mode import BenchmarkServerMode


GITHUB_SERVER_DIR = f'{BENCHMARK_LOCAL_SERVERS}/github-mcp'
ATLAS_WEATHER_DIR = f'{BENCHMARK_LOCAL_SERVERS}/atlas-weather'
PUBMED_SERVER_DIR = f'{BENCHMARK_LOCAL_SERVERS}/pubmed'
WIKIPEDIA_SERVER_DIR = f'{BENCHMARK_LOCAL_SERVERS}/wikipedia-mcp'
ARXIV_LATEX_DIR = f'{BENCHMARK_LOCAL_SERVERS}/arxiv-latex-mcp'
FETCH_SERVER_SCRIPT = f'{BENCHMARK_IMAGE_HELPERS}/fetch_server.py'
ATLAS_ANILI_SERVER_SCRIPT = f'{BENCHMARK_IMAGE_HELPERS}/atlas_anili_server.py'
ATLAS_F1_SERVER_SCRIPT = f'{BENCHMARK_IMAGE_HELPERS}/atlas_f1_server.py'
ATLAS_API_MIMIC_SERVER_SCRIPT = f'{BENCHMARK_IMAGE_HELPERS}/atlas_api_mimic_server.py'
UNIVERSE_BLENDER_CLI_SCRIPT = f'{BENCHMARK_IMAGE_HELPERS}/universe_blender_cli.py'
UNIVERSE_BLENDER_SERVER_SCRIPT = f'{BENCHMARK_IMAGE_HELPERS}/universe_blender_server.py'
TOOLATHLON_EMAILS_SERVER_SCRIPT = f'{BENCHMARK_IMAGE_HELPERS}/toolathlon_emails_server.py'
TOOLATHLON_GOOGLE_CALENDAR_SERVER_SCRIPT = f'{BENCHMARK_IMAGE_HELPERS}/toolathlon_google_calendar_server.py'
TOOLATHLON_12306_SERVER_SCRIPT = f'{BENCHMARK_IMAGE_HELPERS}/toolathlon_12306_server.py'
TOOLATHLON_GOOGLE_FORMS_SERVER_SCRIPT = f'{BENCHMARK_IMAGE_HELPERS}/toolathlon_google_forms_server.py'
TOOLATHLON_GOOGLE_SHEET_SERVER_SCRIPT = f'{BENCHMARK_IMAGE_HELPERS}/toolathlon_google_sheet_server.py'
TOOLATHLON_YOUTUBE_SERVER_SCRIPT = f'{BENCHMARK_IMAGE_HELPERS}/toolathlon_youtube_server.py'
TOOLATHLON_SNOWFLAKE_SERVER_SCRIPT = f'{BENCHMARK_IMAGE_HELPERS}/toolathlon_snowflake_server.py'
TOOLATHLON_K8S_SERVER_SCRIPT = f'{BENCHMARK_IMAGE_HELPERS}/toolathlon_k8s_server.py'
# Nix-installed Chromium has hardcoded rpath and works inside gongfeng sandbox
# containers where LD_LIBRARY_PATH is cleared (breaking /ms-playwright Chrome).
NIX_CHROMIUM_PATH = '/nix/runtime/nix-env/bin/chromium'

TOOLATHLON_SERVER_FILE_NAMES = {
    'arxiv-latex': 'arxiv-latex-mcp',
    'fetch': 'npx-fetch',
    'rail_12306': '12306',
    'scholarly': 'scholarly_search',
    'youtube-transcript': 'youtube_transcript',
}


def npm_bin(name: str) -> str:
    return f'{BENCHMARK_IMAGE_NPM_BIN}/{name}'


PYTHON_BIN = f'{BENCHMARK_IMAGE_VENV_BIN}/python'


def venv_bin(name: str) -> str:
    return f'{BENCHMARK_IMAGE_VENV_BIN}/{name}'


def py311_bin(name: str) -> str:
    return f'{BENCHMARK_IMAGE_PY311_BIN}/{name}'


def py312_bin(name: str) -> str:
    return f'{BENCHMARK_IMAGE_PY312_BIN}/{name}'


def py313_bin(name: str) -> str:
    return f'{BENCHMARK_IMAGE_PY313_BIN}/{name}'


UNIVERSE_PYTHON_ENV = {'PYTHONPATH': UNIVERSE_IMAGE_ROOT}
TOOLATHLON_PYTHON_ENV = {'PYTHONPATH': TOOLATHLON_IMAGE_ROOT}
ATLAS_PUBMED_ENV = {'PYTHONPATH': PUBMED_SERVER_DIR}
ATLAS_WIKIPEDIA_ENV = {'PYTHONPATH': WIKIPEDIA_SERVER_DIR}

# Corporate HTTP proxy used by gongfeng sandbox containers.
_GONGFENG_PROXY = 'http://hyrl-proxy.woa.com:3128'

# Node.js npm module root inside the container image.
_NPM_MODULES = f'{BENCHMARK_IMAGE_NPM_ROOT}/node_modules'

# Proxy env vars for Node.js MCP servers.  NODE_EXTRA_CA_CERTS adds the
# Gongfeng Squid Proxy CA so that TLS through the corporate MITM proxy succeeds.
_NODE_PROXY_ENV: dict[str, str] = {
    'HTTP_PROXY': _GONGFENG_PROXY,
    'HTTPS_PROXY': _GONGFENG_PROXY,
    'NO_PROXY': '.woa.com,localhost,127.0.0.1',
    'NODE_EXTRA_CA_CERTS': '/etc/ssl/certs/ca-certificates.crt',
    # undici is installed at the top-level npm dir; NODE_PATH lets
    # ``require("undici")`` succeed regardless of the process CWD.
    'NODE_PATH': f'{_NPM_MODULES}',
}


# Proxy env vars for Python HTTP clients (requests, httpx, urllib3).
# These libraries read HTTP_PROXY/HTTPS_PROXY automatically.
_PYTHON_PROXY_ENV: dict[str, str] = {
    'HTTP_PROXY': _GONGFENG_PROXY,
    'HTTPS_PROXY': _GONGFENG_PROXY,
    'NO_PROXY': '.woa.com,localhost,127.0.0.1',
    'SSL_CERT_FILE': '/etc/ssl/certs/ca-certificates.crt',
    'REQUESTS_CA_BUNDLE': '/etc/ssl/certs/ca-certificates.crt',
    'HTTPLIB2_CA_CERTS': '/etc/ssl/certs/ca-certificates.crt',
}


_UNDICI_BOOTSTRAP = (
    # Patch native fetch() via undici EnvHttpProxyAgent so that globalThis.fetch
    # honours HTTP_PROXY / HTTPS_PROXY environment variables.
    'try{const u=require("undici");u.setGlobalDispatcher(new u.EnvHttpProxyAgent());}catch(e){}'
)


def _node_proxy_command(npm_package: str, entry: str = 'dist/index.js') -> dict[str, Any]:
    """Build a localize spec that starts a Node.js MCP server with proxy support.

    global-agent crashes on the current runtime image (LD_LIBRARY_PATH=""),
    so we use Node.js built-in undici EnvHttpProxyAgent instead.  A tiny inline
    bootstrap calls ``setGlobalDispatcher(new EnvHttpProxyAgent())`` before
    requiring the real server entry point, making native ``fetch()`` honour
    HTTP_PROXY / HTTPS_PROXY environment variables.
    """
    module_path = f'{_NPM_MODULES}/{npm_package}/{entry}'
    return {
        'command': 'node',
        'args': [
            '-e',
            f'{_UNDICI_BOOTSTRAP};require("{module_path}")',
        ],
        'env': _NODE_PROXY_ENV,
    }


def _node_proxy_command_abs(abs_path: str) -> dict[str, Any]:
    """Like _node_proxy_command but takes an absolute path to a JS entry file."""
    return {
        'command': 'node',
        'args': [
            '-e',
            f'{_UNDICI_BOOTSTRAP};require("{abs_path}")',
        ],
        'env': _NODE_PROXY_ENV,
    }


def _google_maps_esm_command() -> dict[str, Any]:
    """Build a localize spec for @modelcontextprotocol/server-google-maps.

    The package's dist/index.js is pure ESM and opens with::

        import fetch from "node-fetch";

    Module._load monkey-patching (CJS only) does NOT intercept ESM ``import``
    statements, so the node-fetch shim never fires.  We use a different approach:

    1. ``sed -i`` removes the ``import fetch from "node-fetch"`` line (idempotent)
       so the module falls back to Node 20's built-in ``globalThis.fetch``.
    2. A tiny CJS file is written to ``/tmp`` and loaded via ``node --require``
       *before* the ESM entry point, calling ``undici.setGlobalDispatcher``
       so the built-in fetch honours HTTP_PROXY / HTTPS_PROXY.
    """
    dist_path = f'{_NPM_MODULES}/@modelcontextprotocol/server-google-maps/dist/index.js'
    cjs_file = '/tmp/_undici_bootstrap_gmaps.cjs'
    undici_snippet = 'try{const u=require("undici");u.setGlobalDispatcher(new u.EnvHttpProxyAgent())}catch(e){}'
    return {
        'command': 'bash',
        'args': [
            '-c',
            f"sed -i 's/import fetch from .node-fetch.;//' {dist_path}"
            f" && printf '%s\\n' '{undici_snippet}' > {cjs_file}"
            f' && exec node --require {cjs_file} {dist_path}',
        ],
        'env': _NODE_PROXY_ENV,
    }


def _google_python_server_command(
    script_path: str,
    *,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a localize spec for a Google API Python MCP server.

    googleapiclient uses httplib2 under the hood, which does NOT read
    SSL_CERT_FILE or REQUESTS_CA_BUNDLE.  The gongfeng Squid proxy injects
    a self-signed CA for MITM, so httplib2 connections fail with an empty
    SSLHandshakeError unless we patch ``httplib2.CA_CERTS`` at startup.

    The server scripts read HTTPLIB2_CA_CERTS from the environment and
    patch ``httplib2.CA_CERTS`` in their service init function.  The
    patched source is uploaded via ``input_files`` (see ``datasets.py``)
    to override the image-bundled version.
    """
    env = dict(_PYTHON_PROXY_ENV)
    if extra_env:
        env.update(extra_env)
    return {
        'command': PYTHON_BIN,
        'args': [script_path, '--mode', 'api'],
        'env': env,
    }


def merge_env(base: dict[str, str] | None, extra: dict[str, str] | None) -> dict[str, str]:
    merged: dict[str, str] = {}
    if base:
        merged.update(base)
    if extra:
        merged.update(extra)
    return merged


def normalize_server_env(server_name: str, env: dict[str, str]) -> dict[str, str]:
    normalized = dict(env)
    if server_name == 'github':
        github_token = (
            normalized.get('GITHUB_TOKEN')
            or normalized.get('GITHUB_PERSONAL_ACCESS_TOKEN')
            or normalized.get('GITHUB_TOKENS', '').split(',', 1)[0].strip()
        )
        if github_token:
            normalized['GITHUB_TOKEN'] = github_token
    # Google MCP servers (google_sheet, google_calendar, google_forms) read
    # GOOGLE_CREDENTIALS_PATH, but the upstream YAML uses CREDENTIALS_PATH.
    # Map the YAML key to the key the server scripts actually check.
    if server_name in ('google_sheet', 'google_calendar', 'google_forms'):
        cred_path = normalized.get('CREDENTIALS_PATH') or normalized.get('GOOGLE_CREDENTIALS_PATH')
        if cred_path:
            normalized['GOOGLE_CREDENTIALS_PATH'] = cred_path
    return normalized


TOOLATHLON_API_COMMANDS: dict[str, dict[str, Any]] = {
    'filesystem': {'command': npm_bin('mcp-server-filesystem'), 'arg_slice': 2},
    'git': {'command': PYTHON_BIN, 'args': ['-m', 'mcp_server_git']},
    'fetch': _node_proxy_command('@tokenizin/mcp-npx-fetch', 'dist/index.js'),
    'emails': {
        'command': PYTHON_BIN,
        'args': [TOOLATHLON_EMAILS_SERVER_SCRIPT],
        'append_original_args': True,
        'arg_slice': 1,
        'env': TOOLATHLON_PYTHON_ENV,
    },
    'excel': {'command': py311_bin('excel-mcp-server'), 'args': ['stdio']},
    'memory': {'command': npm_bin('mcp-server-memory'), 'args': []},
    'pdf-tools': {'command': py312_bin('pdf-tools-mcp'), 'arg_slice': 1},
    'playwright_with_chunk': {
        'command': npm_bin('mcp-server-playwright'),
        'arg_slice': 2,
        'args': ['--executable-path', NIX_CHROMIUM_PATH, '--no-sandbox'],
        'append_original_args': True,
    },
    'pptx': {'command': venv_bin('ppt_mcp_server'), 'args': []},
    'scholarly': {
        'command': 'bash',
        'args': [
            '-c',
            # Fix arxiv search returning irrelevant papers (SubmittedDate → Relevance)
            "sed -i 's/SortCriterion.SubmittedDate/SortCriterion.Relevance/'"
            f' {BENCHMARK_IMAGE_PY311}/lib/python3.11/site-packages/mcp_scholarly/arxiv_search.py'
            f' && exec {py311_bin("mcp-scholarly")}',
        ],
        'env': _PYTHON_PROXY_ENV,
    },
    'terminal': {'command': venv_bin('cli-mcp-server'), 'args': []},
    'time': {'command': PYTHON_BIN, 'args': ['-m', 'mcp_server_time'], 'append_original_args': True, 'arg_slice': 3},
    'wandb': {'command': py311_bin('wandb_mcp_server'), 'args': []},
    'word': {'command': py311_bin('word_mcp_server'), 'args': []},
    'yahoo-finance': {
        'command': PYTHON_BIN,
        'args': ['-m', 'mcpuniverse.mcp.servers.yahoo_finance'],
        'env': UNIVERSE_PYTHON_ENV,
    },
    'arxiv_local': {'command': py311_bin('arxiv-mcp-server'), 'arg_slice': 2},
    'arxiv-latex': {
        'command': PYTHON_BIN,
        'args': [f'{ARXIV_LATEX_DIR}/server/main.py'],
    },
    'google_map': _google_maps_esm_command(),
    'howtocook': {'command': npm_bin('howtocook-mcp'), 'args': []},
    'huggingface': {'command': npm_bin('mcp-remote'), 'arg_slice': 1},
    'k8s': {'command': PYTHON_BIN, 'args': [TOOLATHLON_K8S_SERVER_SCRIPT]},
    'notion': {'command': npm_bin('notion-mcp-server'), 'arg_slice': 2},
    'youtube-transcript': {'command': venv_bin('mcp-youtube-transcript'), 'args': []},
    'github': _node_proxy_command_abs(f'{GITHUB_SERVER_DIR}/dist/cli.js'),
    'google_calendar': _google_python_server_command(
        TOOLATHLON_GOOGLE_CALENDAR_SERVER_SCRIPT,
        extra_env={'GOOGLE_CREDENTIALS_PATH': '${token.google_oauth2_credentials_path}'},
    ),
    'google_forms': _google_python_server_command(
        TOOLATHLON_GOOGLE_FORMS_SERVER_SCRIPT,
        extra_env={'GOOGLE_CREDENTIALS_PATH': '${token.google_oauth2_credentials_path}'},
    ),
    'google_sheet': _google_python_server_command(
        TOOLATHLON_GOOGLE_SHEET_SERVER_SCRIPT,
        extra_env={'GOOGLE_CREDENTIALS_PATH': '${token.google_oauth2_credentials_path}'},
    ),
    'rail_12306': {'command': PYTHON_BIN, 'args': [TOOLATHLON_12306_SERVER_SCRIPT]},
    'snowflake': {'command': PYTHON_BIN, 'args': [TOOLATHLON_SNOWFLAKE_SERVER_SCRIPT, '--mode', 'api']},
    'youtube': {
        'command': PYTHON_BIN,
        'args': [TOOLATHLON_YOUTUBE_SERVER_SCRIPT, '--mode', 'api'],
        'env': {'YOUTUBE_API_KEY': ''},
    },
}

TOOLATHLON_ORIGINAL_COMPAT_COMMANDS: dict[str, dict[str, Any]] = {
    'emails': {
        'command': PYTHON_BIN,
        'args': [TOOLATHLON_EMAILS_SERVER_SCRIPT],
        'append_original_args': True,
        'arg_slice': 1,
        'env': TOOLATHLON_PYTHON_ENV,
    },
    'google_calendar': _google_python_server_command(
        TOOLATHLON_GOOGLE_CALENDAR_SERVER_SCRIPT,
        extra_env={'GOOGLE_CREDENTIALS_PATH': '${token.google_oauth2_credentials_path}'},
    ),
    'google_forms': _google_python_server_command(
        TOOLATHLON_GOOGLE_FORMS_SERVER_SCRIPT,
        extra_env={'GOOGLE_CREDENTIALS_PATH': '${token.google_oauth2_credentials_path}'},
    ),
    'google_sheet': _google_python_server_command(
        TOOLATHLON_GOOGLE_SHEET_SERVER_SCRIPT,
        extra_env={'GOOGLE_CREDENTIALS_PATH': '${token.google_oauth2_credentials_path}'},
    ),
    'k8s': {'command': PYTHON_BIN, 'args': [TOOLATHLON_K8S_SERVER_SCRIPT]},
    'snowflake': {'command': PYTHON_BIN, 'args': [TOOLATHLON_SNOWFLAKE_SERVER_SCRIPT, '--mode', 'api']},
    'youtube': {'command': PYTHON_BIN, 'args': [TOOLATHLON_YOUTUBE_SERVER_SCRIPT, '--mode', 'api']},
}


def _atlas_api_mimic_command(server_name: str) -> dict[str, Any]:
    return {'command': PYTHON_BIN, 'args': [ATLAS_API_MIMIC_SERVER_SCRIPT, '--server', server_name]}


ATLAS_BASE_COMMANDS: dict[str, dict[str, Any]] = {
    'anili': {'command': PYTHON_BIN, 'args': [ATLAS_ANILI_SERVER_SCRIPT]},
    'arxiv': {'command': py311_bin('arxiv-mcp-server'), 'args': []},
    'calculator': {'command': PYTHON_BIN, 'args': ['-m', 'mcp_server_calculator']},
    'cli-mcp-server': {'command': venv_bin('cli-mcp-server'), 'args': []},
    'clinicaltrialsgov-mcp-server': {'command': npm_bin('clinicaltrialsgov-mcp-server'), 'args': []},
    'context7': {'command': npm_bin('context7-mcp'), 'args': []},
    'ddg-search': {'command': venv_bin('duckduckgo-mcp-server'), 'args': []},
    'desktop-commander': {'command': npm_bin('desktop-commander'), 'args': []},
    'f1-mcp-server': {'command': PYTHON_BIN, 'args': [ATLAS_F1_SERVER_SCRIPT]},
    'fetch': {'command': PYTHON_BIN, 'args': [FETCH_SERVER_SCRIPT], 'env': _PYTHON_PROXY_ENV},
    'filesystem': {'command': npm_bin('mcp-server-filesystem'), 'arg_slice': 1},
    'git': {'command': PYTHON_BIN, 'args': ['-m', 'mcp_server_git']},
    'github': _node_proxy_command_abs(f'{GITHUB_SERVER_DIR}/dist/cli.js'),
    'google-maps': _node_proxy_command('@modelcontextprotocol/server-google-maps'),
    'memory': {'command': npm_bin('mcp-server-memory'), 'args': []},
    'mcp-code-executor': {'command': npm_bin('mcp_code_execution_server'), 'args': []},
    'mcp-server-code-runner': {'command': npm_bin('mcp-server-code-runner'), 'args': []},
    'met-museum': {'command': npm_bin('metmuseum-mcp'), 'args': []},
    'mongodb': {
        'command': npm_bin('mongodb-mcp-server'),
        'args': ['--connectionString'],
        'append_original_args': True,
        'arg_slice': 2,
    },
    'notion': {'command': npm_bin('notion-mcp-server'), 'args': []},
    'open-library': {'command': npm_bin('mcp-open-library'), 'args': []},
    'osm-mcp-server': {'command': py313_bin('osm-mcp-server'), 'args': []},
    'pubmed': {'command': PYTHON_BIN, 'args': ['-m', 'pubmed_server'], 'env': ATLAS_PUBMED_ENV},
    'weather': {'command': 'node', 'args': [f'{ATLAS_WEATHER_DIR}/bin/weather.js']},
    'whois': {'command': npm_bin('whois-mcp'), 'args': []},
    'wikipedia': {'command': venv_bin('wikipedia-mcp'), 'args': [], 'env': {}},
}

ATLAS_API_COMMANDS: dict[str, dict[str, Any]] = copy.deepcopy(ATLAS_BASE_COMMANDS)
ATLAS_API_COMMANDS.update({
    'balldontlie': _atlas_api_mimic_command('balldontlie'),
    'rijksmuseum-server': _atlas_api_mimic_command('rijksmuseum-server'),
})

ATLAS_COMMANDS = ATLAS_API_COMMANDS

UNIVERSE_API_COMMANDS: dict[str, dict[str, Any]] = {
    'blender': {'command': PYTHON_BIN, 'args': [UNIVERSE_BLENDER_SERVER_SCRIPT], 'env': {}},
    'date': {'command': PYTHON_BIN, 'args': ['-m', 'mcpuniverse.mcp.servers.date'], 'env': UNIVERSE_PYTHON_ENV},
    'weather': {
        'command': PYTHON_BIN,
        'args': ['-m', 'mcpuniverse.mcp.servers.weather'],
        'env': UNIVERSE_PYTHON_ENV,
    },
    'echo': {'command': PYTHON_BIN, 'args': ['-m', 'mcpuniverse.mcp.servers.echo'], 'env': UNIVERSE_PYTHON_ENV},
    'google-search': {
        'command': PYTHON_BIN,
        'args': ['-m', 'mcpuniverse.mcp.servers.google_search'],
        'env': UNIVERSE_PYTHON_ENV,
    },
    'wikipedia': {
        'command': PYTHON_BIN,
        'args': ['-m', 'mcpuniverse.mcp.servers.wikipedia'],
        'env': UNIVERSE_PYTHON_ENV,
    },
    'google-maps': _node_proxy_command('@modelcontextprotocol/server-google-maps'),
    'fetch': {'command': PYTHON_BIN, 'args': [FETCH_SERVER_SCRIPT], 'env': _PYTHON_PROXY_ENV},
    'github': _node_proxy_command_abs(f'{GITHUB_SERVER_DIR}/dist/cli.js'),
    'google-sheets': {
        'command': PYTHON_BIN,
        'args': ['-m', 'mcpuniverse.mcp.servers.google_sheets'],
        'env': UNIVERSE_PYTHON_ENV,
    },
    'filesystem': {'command': npm_bin('mcp-server-filesystem'), 'arg_slice': 2},
    'calculator': {'command': PYTHON_BIN, 'args': ['-m', 'mcp_server_calculator']},
    'yfinance': {
        'command': PYTHON_BIN,
        'args': ['-m', 'mcpuniverse.mcp.servers.yahoo_finance'],
        'env': UNIVERSE_PYTHON_ENV,
    },
    'notion': {'command': npm_bin('notion-mcp-server'), 'args': []},
    'chrome': {'command': npm_bin('chrome-devtools-mcp'), 'arg_slice': 1},
    'playwright': {
        'command': npm_bin('playwright-mcp'),
        'arg_slice': 1,
        'args': ['--executable-path', NIX_CHROMIUM_PATH, '--no-sandbox'],
        'append_original_args': True,
    },
}


def _apply_spec(config: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(config)
    original_args = list(config.get('args', []))
    config['command'] = spec['command']
    if 'args' in spec:
        config['args'] = list(spec['args'])
        if spec.get('append_original_args'):
            start = int(spec.get('arg_slice', 0))
            config['args'].extend(original_args[start:])
    elif 'arg_slice' in spec:
        config['args'] = original_args[spec['arg_slice'] :]
    else:
        config['args'] = original_args
    config['env'] = merge_env(config.get('env', {}), spec.get('env'))
    if 'cwd' in spec:
        config['cwd'] = spec['cwd']
    return config


def localize_toolathlon_rendered(
    server_name: str,
    rendered: dict[str, Any],
    *,
    server_mode: BenchmarkServerMode = BenchmarkServerMode.API,
) -> dict[str, Any]:
    localized = copy.deepcopy(rendered)
    name = localized.get('name', server_name)
    spec = TOOLATHLON_ORIGINAL_COMPAT_COMMANDS.get(name) or TOOLATHLON_API_COMMANDS.get(name)
    if spec is None:
        return localized
    localized['params'] = _apply_spec(localized['params'], spec)
    localized['params']['env'] = normalize_server_env(
        localized.get('name', server_name), localized['params'].get('env', {})
    )
    return localized


def localize_atlas_server(
    server_name: str,
    server: dict[str, Any],
    *,
    server_mode: BenchmarkServerMode = BenchmarkServerMode.API,
) -> dict[str, Any]:
    command_map = ATLAS_API_COMMANDS
    spec = command_map.get(server_name)
    if spec is None:
        return copy.deepcopy(server)
    localized = _apply_spec(server, spec)
    localized['env'] = normalize_server_env(server_name, localized.get('env', {}))
    return localized


def localize_universe_server(
    server_name: str,
    server: dict[str, Any],
    *,
    server_mode: BenchmarkServerMode = BenchmarkServerMode.API,
) -> dict[str, Any]:
    localized = copy.deepcopy(server)
    command_map = UNIVERSE_API_COMMANDS
    spec = command_map.get(server_name)
    if spec is None:
        return localized
    stdio_spec = {key: value for key, value in spec.items() if key != 'env'}
    localized_stdio = _apply_spec(localized['stdio'], stdio_spec)
    localized_stdio.pop('env', None)
    localized['stdio'] = localized_stdio
    localized['env'] = normalize_server_env(server_name, merge_env(localized.get('env', {}), spec.get('env')))
    return localized


__all__ = [
    'ARXIV_LATEX_DIR',
    'ATLAS_COMMANDS',
    'ATLAS_IMAGE_ROOT',
    'ATLAS_PUBMED_ENV',
    'ATLAS_WEATHER_DIR',
    'ATLAS_WIKIPEDIA_ENV',
    'GITHUB_SERVER_DIR',
    'PUBMED_SERVER_DIR',
    'TOOLATHLON_API_COMMANDS',
    'TOOLATHLON_ORIGINAL_COMPAT_COMMANDS',
    'TOOLATHLON_IMAGE_ROOT',
    'TOOLATHLON_PYTHON_ENV',
    'TOOLATHLON_SERVER_FILE_NAMES',
    'UNIVERSE_API_COMMANDS',
    'UNIVERSE_IMAGE_ROOT',
    'UNIVERSE_PYTHON_ENV',
    'WIKIPEDIA_SERVER_DIR',
    'localize_atlas_server',
    'localize_toolathlon_rendered',
    'localize_universe_server',
    'merge_env',
    'npm_bin',
    'py311_bin',
    'py312_bin',
    'py313_bin',
    'venv_bin',
    'PYTHON_BIN',
]
