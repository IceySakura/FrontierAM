"""Optional Claude Code MCP configuration helpers."""

from __future__ import annotations

import json

from typing import Any


MCPConfigMode = str
"""One of ``cli`` (inline ``--mcp-config``), ``file`` (``~/.claude/mcp.json``) or ``both``."""


def apply_batch_features(
    *,
    extra_cli_args: list[str],
    config_file_patches: list[dict[str, Any]],
    pre_start_scripts: list[dict[str, Any]],
    mcp_config_servers: dict[str, dict[str, Any]] | None = None,
    mcp_config_mode: MCPConfigMode = 'cli',
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(extra_cli_args, config_file_patches, pre_start_scripts)`` with batch features wired in.

    Inputs are never mutated. Nothing is added when every feature is off, so the
    default path is byte-identical to not calling this at all.
    """
    args = list(extra_cli_args)
    patches = list(config_file_patches)
    scripts = list(pre_start_scripts)

    cc_servers = dict(mcp_config_servers or {})
    if not cc_servers:
        return args, patches, scripts

    mode = (mcp_config_mode or 'cli').strip().lower()
    if mode not in ('cli', 'file', 'both'):
        raise ValueError(f'mcp_config_mode must be cli, file or both, got {mcp_config_mode!r}')

    mcp_json = {'mcpServers': cc_servers}
    allowed = ','.join(f'mcp__{name}__*' for name in cc_servers)

    if mode in ('cli', 'both'):
        # argv is passed to Claude Code without a shell (see runtime_setup.py), so the
        # inline JSON survives as a single argv element — quotes are safe.
        args += ['--mcp-config', json.dumps(mcp_json, ensure_ascii=False), '--strict-mcp-config']

    if mode in ('file', 'both'):
        # ``~`` resolves to the runtime user's HOME inside the sandbox (deep-merged).
        # User scope (~/.claude/mcp.json) avoids the project-scope .mcp.json approval gate.
        patches.append({'path': '~/.claude/mcp.json', 'content': mcp_json})

    # Always allowlist the MCP tools so they are usable even without
    # --dangerously-skip-permissions.
    if allowed and not any(arg in ('--allowedTools', '--allowed-tools') for arg in args):
        args += ['--allowedTools', allowed]

    return args, patches, scripts


__all__ = [
    'MCPConfigMode',
    'apply_batch_features',
]
