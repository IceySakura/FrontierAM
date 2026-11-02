from __future__ import annotations

import base64
import json
import re

from pathlib import PurePosixPath
from typing import Any

import yaml

from nexus.core.tool.mcp import MCPClientConfig, MCPServerSpec, MCPTransportKind

from .constants import ATLAS_DATASET_ROOT, TOOLATHLON_DATASET_ROOT, UNIVERSE_DATASET_ROOT
from .localize import (
    TOOLATHLON_SERVER_FILE_NAMES,
    localize_atlas_server,
    localize_toolathlon_rendered,
    localize_universe_server,
)
from .mode import BenchmarkServerMode


_ATLAS_TOOL_NAME_MAPPINGS = {
    'brave_brave_web_search': 'brave-search_brave_web_search',
    'MongoDB_aggregate': 'mongodb_aggregate',
    'MongoDB_collection-schema': 'mongodb_collection-schema',
    'MongoDB_count': 'mongodb_count',
    'MongoDB_find': 'mongodb_find',
    'MongoDB_list-collections': 'mongodb_list-collections',
    'MongoDB_list-databases': 'mongodb_list-databases',
}

_TOOLATHLON_RE = re.compile(r'\$\{([^}]+)\}')
_UNIVERSE_RE = re.compile(r'\{\{\s*([^}]+?)\s*\}\}')
_ATLAS_EXTRA_SERVER_CONFIGS = {
    'anili': {
        'command': 'python3',
        'args': [],
        'env': {},
    },
    'balldontlie': {
        'command': 'python3',
        'args': [],
        'env': {},
    },
    'f1-mcp-server': {
        'command': 'python3',
        'args': [],
        'env': {},
    },
    'rijksmuseum-server': {
        'command': 'python3',
        'args': [],
        'env': {},
    },
}


def _render_string(value: str, mapping: dict[str, str]) -> str:
    def toolathlon_replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return mapping.get(key, match.group(0))

    def universe_replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return mapping.get(key, match.group(0))

    value = _TOOLATHLON_RE.sub(toolathlon_replace, value)
    return _UNIVERSE_RE.sub(universe_replace, value)


def render_value(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _render_string(value, mapping)
    if isinstance(value, list):
        return [render_value(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: render_value(item, mapping) for key, item in value.items()}
    return value


def _ensure_no_placeholders(value: Any) -> None:
    _ensure_placeholders_allowed(value)


def _ensure_placeholders_allowed(value: Any, *, allowed_prefixes: tuple[str, ...] = ()) -> None:
    text = json.dumps(value, ensure_ascii=False)
    unresolved: list[str] = []
    for match in _TOOLATHLON_RE.finditer(text):
        key = match.group(1)
        if any(key.startswith(prefix) for prefix in allowed_prefixes):
            continue
        unresolved.append(key)
    for match in _UNIVERSE_RE.finditer(text):
        key = match.group(1)
        if any(key.startswith(prefix) for prefix in allowed_prefixes):
            continue
        unresolved.append(key)
    if unresolved:
        raise ValueError(f'unresolved placeholders remain: {text}')


def _stdio_to_spec(server_name: str, config: dict[str, Any], runtime_name: str) -> MCPServerSpec:
    client = MCPClientConfig(
        transport=MCPTransportKind.STDIO,
        command=config['command'],
        args=config.get('args', []),
        env=config.get('env', {}) or {},
        cwd=config.get('cwd'),
        timeout_s=config.get('timeout_s'),
    )
    return MCPServerSpec(name=server_name, runtime_name=runtime_name, client=client)


def _mcp_remote_to_direct_http_spec(
    server_name: str,
    args: list[str],
    *,
    timeout_s: float | None = None,
) -> MCPServerSpec | None:
    """Convert an mcp-remote stdio config to a direct streamable HTTP spec.

    mcp-remote proxies a remote MCP server as local stdio. We can bypass it
    by connecting directly via streamable HTTP, avoiding container LD_LIBRARY_PATH
    issues that break Node.js in Nix-based sandboxes.

    Expected args pattern: [url, --header, "Header: Value", ...]
    """
    if not args:
        return None

    url = args[0]
    if not url.startswith('http'):
        return None

    headers: dict[str, str] = {}
    i = 1
    while i < len(args):
        if args[i] == '--header' and i + 1 < len(args):
            header_value = args[i + 1]
            if ':' in header_value:
                key, _, value = header_value.partition(':')
                headers[key.strip()] = value.strip()
            i += 2
        else:
            i += 1

    client = MCPClientConfig(
        transport=MCPTransportKind.STREAMABLE_HTTP,
        url=url,
        headers=headers,
        timeout_s=timeout_s,
    )
    return MCPServerSpec(name=server_name, client=client)


def _embedded_server_text(
    embedded_server_files: dict[str, dict[str, str]],
    relative_path: str,
) -> str:
    entry = embedded_server_files[relative_path]
    encoding = str(entry.get('encoding') or 'base64').strip().lower()
    content = entry.get('content')
    if not isinstance(content, str):
        raise ValueError(f'Invalid embedded Toolathlon server config: {relative_path}')
    if encoding == 'base64':
        return base64.b64decode(content).decode('utf-8')
    if encoding == 'utf8':
        return content
    raise ValueError(f'Unsupported embedded Toolathlon server config encoding: {encoding}')


def _toolathlon_config_path(
    server_name: str,
    *,
    embedded_server_files: dict[str, dict[str, str]] | None = None,
) -> tuple[str, Any]:
    file_name = TOOLATHLON_SERVER_FILE_NAMES.get(server_name, server_name)
    path = TOOLATHLON_DATASET_ROOT / 'configs' / 'mcp_servers' / f'{file_name}.yaml'
    if path.exists():
        return file_name, yaml.safe_load(path.read_text())
    relative_path = f'configs/mcp_servers/{file_name}.yaml'
    if embedded_server_files and relative_path in embedded_server_files:
        return file_name, yaml.safe_load(_embedded_server_text(embedded_server_files, relative_path))
    return file_name, yaml.safe_load(path.read_text())


def _toolathlon_localized_config(
    server_name: str,
    *,
    server_mode: BenchmarkServerMode,
    embedded_server_files: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    _, data = _toolathlon_config_path(server_name, embedded_server_files=embedded_server_files)
    return localize_toolathlon_rendered(server_name, data, server_mode=server_mode)


def load_toolathlon_server_config_texts(
    *,
    server_mode: BenchmarkServerMode,
    embedded_server_files: dict[str, dict[str, str]] | None = None,
) -> dict[str, str]:
    payloads: dict[str, str] = {}
    config_dir = TOOLATHLON_DATASET_ROOT / 'configs' / 'mcp_servers'
    if config_dir.exists():
        for config_path in sorted(config_dir.glob('*.yaml')):
            raw = yaml.safe_load(config_path.read_text())
            server_name = raw.get('name') or config_path.stem
            rendered = localize_toolathlon_rendered(server_name, raw, server_mode=server_mode)
            payloads[f'configs/mcp_servers/{config_path.name}'] = yaml.safe_dump(
                rendered,
                sort_keys=False,
                allow_unicode=True,
            )
    elif embedded_server_files:
        for relative_path in sorted(embedded_server_files):
            raw = yaml.safe_load(_embedded_server_text(embedded_server_files, relative_path))
            server_name = raw.get('name') or PurePosixPath(relative_path).stem
            rendered = localize_toolathlon_rendered(server_name, raw, server_mode=server_mode)
            payloads[relative_path] = yaml.safe_dump(rendered, sort_keys=False, allow_unicode=True)

    for logical_name, file_name in TOOLATHLON_SERVER_FILE_NAMES.items():
        src = f'configs/mcp_servers/{file_name}.yaml'
        dst = f'configs/mcp_servers/{logical_name}.yaml'
        if src in payloads and dst not in payloads:
            payloads[dst] = payloads[src]

    return payloads


def load_toolathlon_server_specs(
    server_names: list[str],
    *,
    runtime_name: str,
    template_values: dict[str, str],
    server_mode: BenchmarkServerMode = BenchmarkServerMode.API,
    allow_runtime_token_placeholders: bool = False,
    embedded_server_files: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for server_name in server_names:
        localized = _toolathlon_localized_config(
            server_name,
            server_mode=server_mode,
            embedded_server_files=embedded_server_files,
        )
        rendered = render_value(localized, template_values)
        _ensure_placeholders_allowed(rendered, allowed_prefixes=('token.',) if allow_runtime_token_placeholders else ())
        params = rendered.get('params', {})
        command = params.get('command', '')
        args = params.get('args', [])

        # mcp-remote wraps a remote HTTP MCP server as local stdio.  Connect
        # directly via streamable HTTP to avoid LD_LIBRARY_PATH / Node issues
        # inside Nix-based sandbox containers.
        direct_spec: MCPServerSpec | None = None
        if 'mcp-remote' in command:
            direct_spec = _mcp_remote_to_direct_http_spec(
                rendered.get('name', server_name),
                args,
                timeout_s=rendered.get('client_session_timeout_seconds'),
            )

        if direct_spec is not None:
            spec = direct_spec
        else:
            spec = _stdio_to_spec(
                rendered.get('name', server_name),
                {
                    'command': command,
                    'args': args,
                    'env': params.get('env', {}) or {},
                    'cwd': params.get('cwd'),
                    'timeout_s': rendered.get('client_session_timeout_seconds'),
                },
                runtime_name,
            )
        specs.append(spec.model_dump(mode='json', exclude_none=True))
    return specs


def atlas_server_names_from_enabled_tools(enabled_tools: list[str]) -> list[str]:
    servers: list[str] = []
    seen: set[str] = set()
    for tool_name in enabled_tools:
        mapped = _ATLAS_TOOL_NAME_MAPPINGS.get(tool_name, tool_name)
        server = mapped.split('_', 1)[0]
        if server not in seen:
            seen.add(server)
            servers.append(server)
    return servers


def load_atlas_server_specs(
    server_names: list[str],
    *,
    runtime_name: str,
    env_values: dict[str, str],
    server_mode: BenchmarkServerMode = BenchmarkServerMode.API,
) -> list[dict[str, Any]]:
    path = (
        ATLAS_DATASET_ROOT / 'services' / 'agent-environment' / 'src' / 'agent_environment' / 'mcp_server_template.json'
    )
    template = json.loads(path.read_text())['mcpServers']
    specs: list[dict[str, Any]] = []

    for server_name in server_names:
        source = template.get(server_name) or _ATLAS_EXTRA_SERVER_CONFIGS.get(server_name)
        if source is None:
            raise KeyError(f'Atlas server template missing config for {server_name}')
        server = render_value(source, env_values)
        localized = localize_atlas_server(server_name, server, server_mode=server_mode)
        _ensure_no_placeholders(localized)
        spec = _stdio_to_spec(server_name, localized, runtime_name)
        specs.append(spec.model_dump(mode='json', exclude_none=True))

    return specs


def universe_server_names(task_data: dict[str, Any]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for server_ref in task_data.get('mcp_servers', []):
        name = server_ref if isinstance(server_ref, str) else server_ref.get('name')
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def load_universe_server_specs(
    task_data: dict[str, Any],
    *,
    runtime_name: str,
    env_values: dict[str, str],
    server_mode: BenchmarkServerMode = BenchmarkServerMode.API,
) -> list[dict[str, Any]]:
    path = UNIVERSE_DATASET_ROOT / 'mcpuniverse' / 'mcp' / 'configs' / 'server_list.json'
    server_list = json.loads(path.read_text())
    specs: list[dict[str, Any]] = []

    for server_ref in task_data.get('mcp_servers', []):
        name = server_ref if isinstance(server_ref, str) else server_ref['name']
        raw = localize_universe_server(name, server_list[name], server_mode=server_mode)
        stdio = render_value(raw['stdio'], env_values)
        env = render_value(raw.get('env', {}), env_values)
        _ensure_no_placeholders({'stdio': stdio, 'env': env})
        spec = _stdio_to_spec(
            name,
            {
                'command': stdio['command'],
                'args': stdio.get('args', []),
                'env': env,
                'cwd': stdio.get('cwd'),
            },
            runtime_name,
        )
        specs.append(spec.model_dump(mode='json', exclude_none=True))

    return specs
