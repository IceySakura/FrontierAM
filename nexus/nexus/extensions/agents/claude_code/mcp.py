"""Claude Code MCP configuration and additive runtime registration."""

from __future__ import annotations

import base64
import json
import logging
import os
import shlex
import sys
import tempfile
import uuid

from pathlib import Path
from typing import Any, Literal

from nexus.core.tool import MCPServerSpec
from nexus.extensions.agents.installed_common import build_runtime_command
from nexus.extensions.agents.installed_common.execution import AgentExecutionTransport
from nexus.utils.platform import is_windows


logger = logging.getLogger(__name__)


def _native_server_config(server: MCPServerSpec) -> dict[str, Any]:
    if server.prefix is not None:
        raise ValueError(f'Claude Code MCP server {server.name!r} does not support Nexus tool prefix overrides')

    client = server.client
    if client.transport.value == 'stdio':
        assert client.command is not None  # Validated by the stdio client model.
        command = client.command
        args = list(client.args)
        if client.cwd:
            if is_windows():
                from nexus.runtime.executors.command import EXECUTABLE_SHELL, _powershell_args

                command = EXECUTABLE_SHELL
                cwd = client.cwd.replace("'", "''")
                args = _powershell_args(
                    f"Set-Location -LiteralPath '{cwd}'; if (-not $?) {{ exit 1 }}; "
                    + build_runtime_command([client.command, *client.args])
                    + '; exit $LASTEXITCODE'
                )
            else:
                command = '/bin/sh'
                args = [
                    '-c',
                    'cd -- "$1" && shift && exec "$@"',
                    'nexus-mcp-cwd',
                    client.cwd,
                    client.command,
                    *client.args,
                ]
        config: dict[str, Any] = {
            'type': 'stdio',
            'command': command,
            'args': args,
        }
        if client.env:
            config['env'] = client.env
    else:
        config = {
            'type': 'sse' if client.transport.value == 'sse' else 'http',
            'url': client.url,
        }
        if client.headers:
            config['headers'] = client.headers
    if client.timeout_s is not None:
        config['timeout'] = max(1, round(client.timeout_s * 1000))
    return config


def mcp_config_servers_from_specs(servers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Convert shared Nexus MCP specs to Claude Code's native server map.

    ``MCPServerSpec`` retains compatibility with the historical flat stdio shape,
    while the native map can also represent HTTP/SSE transports and per-server
    environment variables for ``claude mcp add-json`` or ``--mcp-config``.
    """
    result: dict[str, dict[str, Any]] = {}
    for index, raw_server in enumerate(servers):
        try:
            server = MCPServerSpec.model_validate(raw_server)
        except ValueError as exc:
            raise ValueError(f'Invalid Claude Code MCP server config at index {index}') from exc
        if server.name in result:
            raise ValueError(f'Duplicate Claude Code MCP server name: {server.name!r}')
        result[server.name] = _native_server_config(server)
    return result


async def register_mcp_servers(
    claude_path: str | Path,
    *,
    working_dir: str,
    claude_env: dict[str, str],
    mcp_servers: list[dict[str, Any]] | None,
    scope: Literal['local', 'project', 'user'] = 'local',
    transport: AgentExecutionTransport | None = None,
    invocation_id: str | None = None,
) -> None:
    """Add Nexus MCP specs to Claude's existing local configuration.

    ``add-json`` preserves the old additive ``mcp_servers`` contract while
    supporting every transport representable by the shared Nexus model.
    Explicit strict replacement remains the separate ``mcp_config_servers`` path.
    """
    from nexus.runtime.executors.command import RunCommandRequest, run_command

    resolved_transport = transport or AgentExecutionTransport()
    resolved_invocation_id = invocation_id or 'mcp-registration'
    native_servers = mcp_config_servers_from_specs(mcp_servers or [])
    if not native_servers:
        return
    temp_dir = (
        resolved_transport.shared_path('mcp-files', default=tempfile.gettempdir())
        if resolved_transport.is_local
        else Path(claude_env['HOME']) / '.local/share/nexus/mcp'
    )
    await resolved_transport.mkdir(temp_dir)
    for server_name, config in native_servers.items():
        config_path: Path | None = None
        try:
            if resolved_transport.is_local:
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    encoding='utf-8',
                    prefix='.nexus_mcp_',
                    suffix='.json',
                    dir=str(temp_dir),
                    delete=False,
                ) as config_file:
                    json.dump(config, config_file, ensure_ascii=False, separators=(',', ':'))
                    config_path = Path(config_file.name)
                os.chmod(config_path, 0o600)
            else:
                config_path = temp_dir / f'{uuid.uuid4().hex}.json'
                await resolved_transport.write_text(config_path, json.dumps(config, ensure_ascii=False))

            command_parts = [shlex.quote(str(claude_path)), 'mcp', 'add-json']
            if scope != 'local':
                command_parts.extend(['--scope', shlex.quote(scope)])
            command_parts.extend([
                shlex.quote(server_name),
                f'"$(cat -- {shlex.quote(str(config_path))})"',
            ])
            command = ' '.join(command_parts)
            if resolved_transport.is_local and is_windows():
                # Windows PowerShell strips native argv quotes from JSON. Pass it
                # through Python subprocess instead; keep secrets in the temp file.
                argv = [str(claude_path), 'mcp', 'add-json']
                if scope != 'local':
                    argv.extend(['--scope', scope])
                code = (
                    'import pathlib, subprocess, sys; '
                    f'config = pathlib.Path({str(config_path)!r}).read_text(encoding="utf-8"); '
                    f'argv = {argv!r} + [{server_name!r}, config]; '
                    'sys.exit(subprocess.call(argv))'
                )
                encoded = base64.b64encode(code.encode('utf-8')).decode('ascii')
                command = build_runtime_command([
                    sys.executable,
                    '-c',
                    f"import base64; exec(base64.b64decode('{encoded}'))",
                ])
            logger.debug('Registering MCP server %s (type=%s)', server_name, config.get('type', 'unknown'))
            if resolved_transport.is_local:
                result = await run_command(
                    RunCommandRequest(
                        command=command,
                        cwd=working_dir,
                        timeout=30,
                        extra_env=claude_env,
                    )
                )
            else:
                shell = await resolved_transport.find_binary('bash')
                async with resolved_transport.prepare(
                    [shell, '-c', command],
                    working_dir=working_dir,
                    invocation_id=resolved_invocation_id,
                    env=claude_env,
                ) as prepared:
                    result = await run_command(
                        RunCommandRequest(
                            command=shlex.join(prepared.command),
                            cwd=prepared.cwd,
                            timeout=30,
                            extra_env=prepared.env,
                        )
                    )
        finally:
            if config_path is not None:
                await resolved_transport.remove(config_path)

        if result.return_code != 0:
            raise RuntimeError(
                f'MCP add-json {server_name!r} failed with return code {result.return_code}: '
                f'{(result.stderr or result.stdout or "no output").strip()}'
            )
        logger.debug('MCP server %s registered', server_name)


__all__ = ['mcp_config_servers_from_specs', 'register_mcp_servers']
