"""Software-name → MCP server catalog for Claude Code native-bridge rollouts.

A rollout config can name the *software* it needs (``mcp_apps: [blender, kicad]``) instead of
spelling out each MCP server's ``command``/``args``. This module maps a software name to the MCP
server spec, and the resolved specs feed the existing ``mcp_servers`` staging path
(``ClaudeCodeNativeBridgeInput.mcp_servers`` → ``_write_mcp_config`` → ``claude --mcp-config``).

The catalog only *registers* servers with Claude Code; it does not install the server code. The
launch commands below point at ``/home/user/mcp/*``, matching the creator-apps OSWorld image where
those servers are baked in. A config whose image lays the servers out differently can bypass the
catalog and pass full specs via ``mcp_servers`` directly.

Seed values were read verbatim from a live instance of that image (``claude mcp list`` +
``~/.claude.json``): blender, freecad, godot, kicad, figma, playwright, unreal.
"""

from __future__ import annotations

from typing import Any


# name → MCP server spec, in the {name, command, args} shape the native-bridge staging expects
# (``_write_mcp_config`` copies every key except ``name`` into the Claude ``mcpServers`` entry, so a
# stdio server needs ``command`` and ``args``). Keep keys lowercase; ``resolve_mcp_apps`` lowercases
# the lookup.
SOFTWARE_MCP_CATALOG: dict[str, dict[str, Any]] = {
    'blender': {
        'name': 'blender',
        'command': '/home/user/mcp/blender-mcp/.venv/bin/blender-mcp',
        'args': [],
    },
    'freecad': {
        'name': 'freecad',
        'command': '/home/user/mcp/freecad-mcp/.venv/bin/freecad-mcp',
        'args': [],
    },
    'godot': {
        'name': 'godot',
        'command': 'node',
        'args': ['/home/user/mcp/godot-mcp/build/index.js'],
    },
    'kicad': {
        'name': 'kicad',
        'command': 'bash',
        'args': ['-lc', 'cd /home/user/mcp/KiCAD-MCP-Server && exec node dist/index.js'],
    },
    'figma': {
        'name': 'figma',
        'command': 'npx',
        'args': ['-y', '--offline', 'figma-developer-mcp', '--figma-api-key=DUMMY_TOKEN', '--stdio'],
    },
    'playwright': {
        'name': 'playwright',
        'command': 'npx',
        'args': ['-y', '--offline', '@playwright/mcp@0.0.78', '--headless', '--browser', 'chrome'],
    },
    'unreal': {
        'name': 'unreal',
        'command': 'node',
        'args': ['/home/user/mcp/Unreal_mcp/dist/cli.js'],
    },
}


def _spec_copy(spec: dict[str, Any]) -> dict[str, Any]:
    """Deep-ish copy of a catalog spec so callers can't mutate the shared catalog."""
    return {key: (list(value) if isinstance(value, list) else value) for key, value in spec.items()}


def resolve_mcp_apps(apps: list[str]) -> list[dict[str, Any]]:
    """Expand software names to MCP server specs.

    Names are matched case-insensitively and de-duplicated (first occurrence wins). An unknown name
    raises ``ValueError`` listing the known keys, so a typo in an explicit config fails fast — before
    the expensive rollout — rather than silently running with a tool missing.
    """
    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for app in apps:
        key = str(app).strip().lower()
        if not key or key in seen:
            continue
        spec = SOFTWARE_MCP_CATALOG.get(key)
        if spec is None:
            known = ', '.join(sorted(SOFTWARE_MCP_CATALOG))
            raise ValueError(f'unknown MCP software {app!r}; known: {known}')
        seen.add(key)
        resolved.append(_spec_copy(spec))
    return resolved


def merge_mcp_servers(
    explicit: list[dict[str, Any]],
    from_apps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge explicit ``mcp_servers`` specs with catalog-resolved ones, de-duped by ``name``.

    An explicit entry wins over a catalog entry of the same name, so a config can override one
    server's command while still naming the rest via ``mcp_apps``. Explicit entries keep their order
    and come first; catalog entries whose name is not already present are appended.
    """
    merged: list[dict[str, Any]] = []
    names: set[str] = set()
    for spec in explicit:
        name = str(spec.get('name') or '').strip()
        if name:
            names.add(name)
        merged.append(spec)
    for spec in from_apps:
        if str(spec.get('name') or '').strip() in names:
            continue
        merged.append(spec)
    return merged


__all__ = ['SOFTWARE_MCP_CATALOG', 'resolve_mcp_apps', 'merge_mcp_servers']
