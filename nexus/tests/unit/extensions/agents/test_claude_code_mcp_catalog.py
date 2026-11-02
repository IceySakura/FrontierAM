"""Tests for the software-name → MCP catalog and its wiring into the native-bridge agent."""

from typing import Any

import pytest

from nexus.extensions.agents.claude_code.agent import (
    ClaudeCodeNativeBridgeInput,
    run_claude_code_native_bridge,
)
from nexus.extensions.agents.claude_code.mcp_catalog import (
    SOFTWARE_MCP_CATALOG,
    merge_mcp_servers,
    resolve_mcp_apps,
)


def test_resolve_known_apps_returns_specs() -> None:
    specs = resolve_mcp_apps(['blender', 'kicad'])
    assert [s['name'] for s in specs] == ['blender', 'kicad']
    blender = specs[0]
    assert blender['command'] == '/home/user/mcp/blender-mcp/.venv/bin/blender-mcp'
    assert blender['args'] == []
    # every catalog entry is a valid stdio spec
    for spec in SOFTWARE_MCP_CATALOG.values():
        assert spec['name']
        assert spec['command']
        assert isinstance(spec['args'], list)


def test_resolve_is_case_insensitive_and_deduped() -> None:
    specs = resolve_mcp_apps(['Blender', ' KICAD ', 'blender'])
    assert [s['name'] for s in specs] == ['blender', 'kicad']


def test_resolve_unknown_app_raises_listing_known() -> None:
    with pytest.raises(ValueError, match='unknown MCP software') as exc:
        resolve_mcp_apps(['bl3nder'])
    # the error names the known keys so a typo is easy to fix
    assert 'blender' in str(exc.value)


def test_resolve_returns_independent_copies() -> None:
    first = resolve_mcp_apps(['godot'])[0]
    first['args'].append('MUTATED')
    assert SOFTWARE_MCP_CATALOG['godot']['args'] == ['/home/user/mcp/godot-mcp/build/index.js']


def test_merge_explicit_wins_by_name() -> None:
    explicit = [{'name': 'blender', 'command': '/custom/blender', 'args': []}]
    merged = merge_mcp_servers(explicit, resolve_mcp_apps(['blender', 'kicad']))
    assert [s['name'] for s in merged] == ['blender', 'kicad']
    # the explicit blender entry is kept; the catalog one is dropped
    assert next(s for s in merged if s['name'] == 'blender')['command'] == '/custom/blender'


@pytest.mark.asyncio
async def test_native_bridge_agent_merges_mcp_apps_into_mcp_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_call_native_bridge_runtime_agent(runtime_agent: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {'success': True, 'output': '', 'stderr': '', 'exit_code': 0, 'execution_time': 1.0}

    monkeypatch.setattr(
        'nexus.extensions.agents.claude_code.agent.call_native_bridge_runtime_agent',
        fake_call_native_bridge_runtime_agent,
    )

    await run_claude_code_native_bridge(
        ClaudeCodeNativeBridgeInput(
            user_prompt='use the blender mcp',
            working_dir='/tmp/work',
            timeout=3600,
            ensure_claude=True,
            mcp_apps=['blender', 'kicad'],
            mcp_servers=[{'name': 'kicad', 'command': '/custom/kicad', 'args': []}],
        )
    )

    servers = captured['mcp_servers']
    assert [s['name'] for s in servers] == ['kicad', 'blender']
    # explicit kicad spec wins over the catalog entry
    assert next(s for s in servers if s['name'] == 'kicad')['command'] == '/custom/kicad'
    # blender came from the catalog
    assert next(s for s in servers if s['name'] == 'blender')['command'] == (
        '/home/user/mcp/blender-mcp/.venv/bin/blender-mcp'
    )


@pytest.mark.asyncio
async def test_native_bridge_agent_unknown_app_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_call_native_bridge_runtime_agent(runtime_agent: Any, **kwargs: Any) -> dict[str, Any]:
        return {'success': True, 'output': '', 'stderr': '', 'exit_code': 0, 'execution_time': 1.0}

    monkeypatch.setattr(
        'nexus.extensions.agents.claude_code.agent.call_native_bridge_runtime_agent',
        fake_call_native_bridge_runtime_agent,
    )

    with pytest.raises(ValueError, match='unknown MCP software'):
        await run_claude_code_native_bridge(
            ClaudeCodeNativeBridgeInput(user_prompt='x', working_dir='/tmp/work', mcp_apps=['nope']),
        )
