"""Reusable run parts for Claude Code."""

from __future__ import annotations

from typing import Any

from nexus.runner.compose import run_part


@run_part('claude_code')
def claude_code_run_part(params: dict[str, Any]) -> dict[str, Any]:
    if params:
        raise ValueError(f'Run part claude_code does not accept parameters: {sorted(params)}')
    return {
        'agent_name': 'claude_code',
        'agent_config': {
            'version': '2.1.153',
            'npm_registry': 'https://registry.npmmirror.com/',
            'extra_cli_args': ['--output-format', 'json', '--verbose'],
        },
    }
