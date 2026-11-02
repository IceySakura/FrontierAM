"""Codex config, environment, and command helpers for native bridge runs."""

from __future__ import annotations

import json
import re

from nexus.extensions.agents.codex.native_bridge.adapters.types import CodexRunRequest
from nexus.extensions.agents.installed_common import DEFAULT_PROXY_API_KEY, splice_toml_root_keys
from nexus.utils.shell import quote_posix


# Root-level `model_instructions_file = ...` assignment inside extra_config_toml;
# the task-staged file replaces it instead of colliding into invalid TOML.
# Key forms cover bare, "double-quoted", and 'single-quoted' keys (all legal
# TOML). Occurrences *inside* a multi-line string value (e.g. example lines
# embedded in developer_instructions) must survive: a line-anchored regex
# cannot tell a root key from a continuation line of a string value, so the
# strip runs as a scanner that tracks multi-line delimiters and only removes
# top-level keys -- a key whose value spans a multi-line string is removed
# together with that value.
_ROOT_MODEL_INSTRUCTIONS_KEY_RE = re.compile(
    r'[ \t]*(?:"model_instructions_file"|\'model_instructions_file\'|model_instructions_file)[ \t]*=[ \t]*'
)


def _strip_root_model_instructions_key(extra_config_toml: str) -> str:
    """Remove root-level ``model_instructions_file`` assignments.

    Lines inside an unterminated multi-line string value (``\"\"\"``/``'''``)
    are passed through untouched, so a same-named example line inside a
    multi-line ``developer_instructions`` value survives; a key whose own
    value opens an unterminated multi-line string is dropped together with
    the rest of that value.
    """
    out: list[str] = []
    in_ml: str | None = None  # active multi-line delimiter while inside a value
    dropping = False  # True while the tracked value belongs to the removed key
    for line in extra_config_toml.splitlines(keepends=True):
        if in_ml is not None:
            if not dropping:
                out.append(line)
            if line.count(in_ml) % 2 == 1:
                in_ml = None
                dropping = False
            continue
        if _ROOT_MODEL_INSTRUCTIONS_KEY_RE.match(line):
            for delim in ('"""', "'''"):
                if line.count(delim) % 2 == 1:
                    in_ml = delim
                    dropping = True
                    break
            continue
        out.append(line)
        for delim in ('"""', "'''"):
            if line.count(delim) % 2 == 1:
                in_ml = delim
                break
    return ''.join(out)


def build_codex_config(request: CodexRunRequest, codex_home: str = '') -> str:
    lines: list[str] = []
    if request.model:
        lines.append(f'model = {json.dumps(request.model)}')
    if request.auto_compact_token_limit is not None:
        lines.append(f'model_auto_compact_token_limit = {int(request.auto_compact_token_limit)}')
    if request.model_catalog_json and codex_home:
        lines.append(f'model_catalog_json = {json.dumps(codex_home.rstrip("/") + "/model_catalog.json")}')
    if request.model_instructions_file:
        # Root-level key; emitted here, ahead of every table header, so it
        # cannot be scoped into a table.
        lines.append(f'model_instructions_file = {json.dumps(request.model_instructions_file)}')
    provider_name = 'azure' if request.use_azure_provider else 'Nexus Proxy'
    lines.extend([
        'model_provider = "nexus"',
        '',
        '[model_providers.nexus]',
        f'name = {json.dumps(provider_name)}',
        f'base_url = {json.dumps(request.launch.local_base_url.rstrip("/") + "/v1")}',
        'env_key = "NEXUS_PROXY_API_KEY"',
        'wire_api = "responses"',
    ])
    for server in request.mcp_servers or []:
        name = str(server.get('name') or '')
        command = str(server.get('command') or '')
        if not name or not command:
            raise ValueError('Each Codex MCP server requires non-empty name and command fields')
        args_value = server.get('args') or []
        if not isinstance(args_value, list):
            raise ValueError(f'Codex MCP server {name!r} args must be a list')
        args = [str(arg) for arg in args_value]
        lines.extend([
            '',
            f'[mcp_servers.{json.dumps(name)}]',
            f'command = {json.dumps(command)}',
            f'args = {json.dumps(args)}',
        ])
        env = server.get('env')
        if env:
            if not isinstance(env, dict):
                raise ValueError(f'Codex MCP server {name!r} env must be a mapping')
            lines.append(
                'env = {'
                + ', '.join(f'{json.dumps(str(key))} = {json.dumps(str(value))}' for key, value in env.items())
                + '}'
            )
        for key in ('cwd', 'startup_timeout_sec', 'tool_timeout_sec', 'enabled', 'required'):
            if key in server:
                lines.append(f'{key} = {json.dumps(server[key])}')
        for key in ('enabled_tools', 'disabled_tools', 'omit_tools_from'):
            if key not in server:
                continue
            tools = server[key]
            if not isinstance(tools, list):
                raise ValueError(f'Codex MCP server {name!r} {key} must be a list')
            lines.append(f'{key} = {json.dumps([str(tool) for tool in tools])}')
    extra_config = (request.extra_config_toml or '').strip()
    if request.model_instructions_file and extra_config:
        # The task-staged file is this run's contract; the same-named key is
        # removed from extra_config_toml (not left in place) -- two identical
        # keys, however equal their values, make the TOML invalid.
        extra_config = _strip_root_model_instructions_key(extra_config).strip()
    lines = splice_toml_root_keys(lines, extra_config)
    return '\n'.join(lines) + '\n'


def build_codex_env(*, codex_home: str, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        'CODEX_HOME': codex_home,
        'NEXUS_PROXY_API_KEY': DEFAULT_PROXY_API_KEY,
        'IS_SANDBOX': '1',
        'NO_COLOR': '1',
    }
    env.update(extra_env or {})
    return env


def build_codex_command(request: CodexRunRequest, *, output_file: str) -> list[str]:
    prompt = request.task
    if request.developer_instructions:
        prompt = f'{request.developer_instructions.rstrip()}\n\n{prompt}'
    # Deliberately NOT passing --ephemeral, same as the non-bridge path: multi_agent_version=v2
    # spawns a subagent by reconstructing the parent context from the persistent thread store, and
    # without it load_agent_model_context fails with `collab spawn failed: no thread with id: <id>`.
    # The sessions this leaves behind land in the per-run codex-home, which is thrown away anyway.
    command = [
        request.codex_executable,
        'exec',
        '--dangerously-bypass-approvals-and-sandbox',
        '--skip-git-repo-check',
    ]
    if request.emit_json_trace:
        command.append('--json')
    command.extend(request.extra_cli_args or [])
    command.extend(['-C', request.working_dir, '-o', output_file, prompt])
    return command


def shell_join_env(env: dict[str, str]) -> str:
    return ' '.join(f'{key}={quote_posix(value)}' for key, value in env.items())


def shell_join_cmd(command: list[str]) -> str:
    return ' '.join(quote_posix(part) for part in command)


def build_pre_start_command(interpreter: str, code: str, *, cwd: str, env: dict[str, str]) -> str:
    if interpreter == 'bash':
        inner = f'bash -c {quote_posix(code)}'
    elif interpreter == 'python':
        inner = f'python3 -c {quote_posix(code)}'
    else:
        raise ValueError(f'Unsupported pre-start interpreter: {interpreter}')
    quoted_cwd = quote_posix(cwd)
    return ' '.join([
        f'mkdir -p {quoted_cwd}',
        '&&',
        f'cd {quoted_cwd}',
        '&&',
        f'{shell_join_env(env)} {inner}',
    ])


__all__ = [
    'build_codex_command',
    'build_codex_config',
    'build_codex_env',
    'build_pre_start_command',
    'shell_join_cmd',
    'shell_join_env',
]
