"""Resolving an MCP server's interpreter, inside the sandbox.

A stdio MCP server is a command someone else runs later -- ``claude mcp add`` records
it and the worker's Claude Code spawns it; ``mcp_start_proxy`` spawns it in a worker
process -- so a bare ``python`` resolves against that process's ``PATH`` rather than
against whatever registered it. The interpreter running this module is the one to use,
and a bare command is replaced with it.

Which interpreter that is matters, measured in a sandbox on 0821:

===========================================  ===========  ==============
interpreter                                  mcp, httpx   hitl package
===========================================  ===========  ==============
``/nix/runtime/nexus/.venv-3.11`` (base)     yes          **no**
``/tmp/nexus/runtime/venvs/<id>`` (worker)   yes          yes
===========================================  ===========  ==============

Both ``runtime.invoke`` and an executor call land in the worker venv when a run
configures ``worker_venv``, which is the one carrying this package -- so a server
declared as ``python -m nexus...`` works, and the same entry naming the base venv by
absolute path fails with ``ModuleNotFoundError`` after the session is already open.

Runs sandbox-side. The host cannot do this: it does not know which venv the sandbox
built, and the entry it declares travels as data.
"""

import logging
import sys

from typing import Any


logger = logging.getLogger(__name__)


BARE_INTERPRETERS = frozenset({'python', 'python3'})
"""Commands replaced with this process's interpreter."""


def resolve_interpreters(mcp_servers: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Every entry, with a bare interpreter command replaced by an absolute one.

    Reads both shapes: the worker's flat ``{name, command, args}`` and the user's
    nested ``{name, client: {command, ...}}``. An entry naming anything else is passed
    through -- an absolute path is a choice, and a server that is not a Python program
    has nothing to resolve.

    The input is not mutated: the caller's copy is a run's config, which other code
    reads afterwards.
    """
    resolved: list[dict[str, Any]] = []
    for entry in mcp_servers or []:
        name = entry.get('name', '(unnamed)')
        client = entry.get('client')
        if isinstance(client, dict):
            resolved.append({**entry, 'client': {**client, **_resolved_command(client, name=name)}})
            continue
        resolved.append({**entry, **_resolved_command(entry, name=name)})
    return resolved


def _resolved_command(holder: dict[str, Any], *, name: object) -> dict[str, Any]:
    """``{'command': ...}`` when ``holder`` names a bare interpreter, else ``{}``."""
    command = holder.get('command')
    if not isinstance(command, str) or command not in BARE_INTERPRETERS:
        return {}
    logger.info('MCP server %s: resolving %s to %s', name, command, sys.executable)
    return {'command': sys.executable}


async def resolve_in_sandbox(runtime: Any, mcp_servers: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """:func:`resolve_interpreters`, run in the sandbox rather than on the host.

    For the user's own servers, which the host declares and ``load_mcp_tools`` starts
    through a proxy: the host's interpreter is not the sandbox's, so resolving here
    would name a path that does not exist there. ``runtime.invoke`` lands in the venv
    the run built, which is the one carrying this package.

    Returns the entries unchanged when nothing needs resolving, so a caller can hand
    over whatever it has.
    """
    if not mcp_servers:
        return []
    if not any(_needs_resolving(entry) for entry in mcp_servers):
        return [dict(entry) for entry in mcp_servers]
    return await runtime.invoke(resolve_interpreters, mcp_servers)


def _needs_resolving(entry: dict[str, Any]) -> bool:
    """Whether this entry names a bare interpreter, in either shape."""
    client = entry.get('client')
    holder = client if isinstance(client, dict) else entry
    return holder.get('command') in BARE_INTERPRETERS


__all__ = ['BARE_INTERPRETERS', 'resolve_in_sandbox', 'resolve_interpreters']
