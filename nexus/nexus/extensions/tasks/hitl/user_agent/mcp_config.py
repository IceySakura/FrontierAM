"""MCP servers a run declares, for the user simulator and for the worker.

The two sides consume different shapes, so they are declared separately rather
than derived from one another:

* the **user** side hands specs to ``ExecutionContext.load_mcp_tools``, which
  takes ``{name, runtime_name, client: {...}}`` and honours ``client.env``;
* the **worker** side reaches Claude Code through ``claude mcp add``, which reads
  only ``name`` / ``command`` / ``args``
  (``actor/print_claude_code/runtime.py:477``). It has no ``--env``: the server
  process inherits the worker's environment, so its variables belong in
  ``worker_config['extra_env']``.

A worker entry written in the user shape is not an error Claude Code reports --
``command`` reads as empty, the registration is skipped with a log line, and the
run proceeds without the server. Both shapes are therefore checked here, where a
mistake stops the batch.

The user shape follows ``aidata``'s (``tasks/aidata/case/spec.py``): flat for
brevity, nested when a transport needs it, ``runtime_name`` defaulting to the
sandbox.
"""

import json
import logging
import os
import re

from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


USER_ENV = 'HITL_USER_MCP_SERVERS'
WORKER_ENV = 'HITL_WORKER_MCP_SERVERS'

DEFAULT_RUNTIME = 'main'
"""Where a user-side server runs unless the declaration says otherwise.

The connection is made from inside the sandbox through a proxy, so a streaming
response is parsed by the runtime rather than the host. A server only the host
can reach needs an explicit ``"runtime_name": null``.
"""

WORKER_KEYS = frozenset({'name', 'command', 'args'})

_PLACEHOLDER = re.compile(r'\$\{[^}]+\}|\$[A-Z_][A-Z0-9_]*')


class MCPConfigError(ValueError):
    """A declared MCP server cannot be used as written."""


def _reject_placeholders(entry: Any, *, source: str, name: str) -> None:
    """Refuse ``${VAR}`` anywhere in an entry.

    Nothing expands it: the stdio transport merges ``client.env`` into the
    environment as given (``core/tool/mcp/transport/factory.py:47-64``), so the
    server would receive the literal text and fail on a credential that looks
    present.
    """
    text = json.dumps(entry, ensure_ascii=False)
    found = sorted({match.group(0) for match in _PLACEHOLDER.finditer(text)})
    if found:
        raise MCPConfigError(
            f'{source}: server {name!r} contains {found}, which nothing expands. '
            'Name the variable in the launch script instead, or let the generator fill it in.'
        )


def normalise_user_entry(raw: Any, *, source: str) -> dict[str, Any]:
    """One user-side entry as ``load_mcp_tools`` takes it.

    Accepts the nested form, a ``url`` for an HTTP transport, or the flat stdio
    form. ``runtime_name`` is defaulted rather than required, and an explicit
    ``null`` is kept.
    """
    if not isinstance(raw, dict) or not raw.get('name'):
        raise MCPConfigError(f'{source}: every entry must be an object with a name, got {raw!r}')
    name = str(raw['name'])
    _reject_placeholders(raw, source=source, name=name)
    runtime_name = raw.get('runtime_name', DEFAULT_RUNTIME)

    if 'client' in raw:
        client = raw['client']
        if not isinstance(client, dict):
            raise MCPConfigError(f'{source}: server {name!r} has a client that is not an object')
    elif 'url' in raw:
        client = {
            'transport': raw.get('transport', 'aiohttp_streamable_http'),
            'url': raw['url'],
            'headers': dict(raw.get('headers') or {}),
        }
        if raw.get('timeout_s') is not None:
            client['timeout_s'] = float(raw['timeout_s'])
    elif 'command' in raw:
        client = {
            'transport': 'stdio',
            'command': raw['command'],
            'args': list(raw.get('args') or []),
            'env': dict(raw.get('env') or {}),
        }
    else:
        raise MCPConfigError(f'{source}: server {name!r} needs a client, a url, or a command')

    entry: dict[str, Any] = {'name': name, 'client': client}
    if runtime_name is not None:
        entry['runtime_name'] = runtime_name
    if raw.get('prefix'):
        entry['prefix'] = raw['prefix']
    return entry


def check_worker_entry(raw: Any, *, source: str) -> dict[str, Any]:
    """One worker-side entry, in the only shape ``claude mcp add`` reads.

    Refuses the user shape and refuses ``env``, both of which would otherwise
    leave the server unregistered or unconfigured with nothing said.
    """
    if not isinstance(raw, dict) or not raw.get('name'):
        raise MCPConfigError(f'{source}: every entry must be an object with a name, got {raw!r}')
    name = str(raw['name'])
    _reject_placeholders(raw, source=source, name=name)

    if 'client' in raw:
        raise MCPConfigError(
            f'{source}: server {name!r} uses the nested client form, which Claude Code does not read. '
            'Give name, command and args at the top level.'
        )
    if 'env' in raw:
        raise MCPConfigError(
            f'{source}: server {name!r} declares env, which `claude mcp add` cannot pass. '
            'Put those variables in the worker environment instead.'
        )
    if not raw.get('command'):
        raise MCPConfigError(f'{source}: server {name!r} needs a command')

    unknown = sorted(set(raw) - WORKER_KEYS)
    if unknown:
        raise MCPConfigError(
            f'{source}: server {name!r} declares {unknown}, which is ignored here. '
            f'Only {sorted(WORKER_KEYS)} reach Claude Code.'
        )
    return {'name': name, 'command': raw['command'], 'args': list(raw.get('args') or [])}


def _load(env_key: str) -> list[Any]:
    """The JSON array a run pointed ``env_key`` at, or empty when unset.

    A path naming no file raises: returning nothing would run the whole batch
    without the servers and say so nowhere.
    """
    raw = os.environ.get(env_key, '').strip()
    if not raw:
        return []
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise MCPConfigError(f'{env_key} names no file: {path}')
    try:
        loaded = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise MCPConfigError(f'{env_key} ({path}) is not valid JSON: {exc}') from exc
    if not isinstance(loaded, list):
        raise MCPConfigError(f'{env_key} ({path}) must hold a list of servers, got {type(loaded).__name__}')
    return loaded


def _duplicates(names: list[str]) -> list[str]:
    return sorted({name for name in names if names.count(name) > 1})


def user_servers() -> list[dict[str, Any]]:
    """The user simulator's declared servers, ready for ``load_mcp_tools``."""
    entries = [normalise_user_entry(raw, source=USER_ENV) for raw in _load(USER_ENV)]
    repeated = _duplicates([entry['name'] for entry in entries])
    if repeated:
        raise MCPConfigError(f'{USER_ENV}: servers declared more than once: {repeated}')
    return entries


def worker_servers() -> list[dict[str, Any]]:
    """The worker's declared servers, in ``claude mcp add`` shape."""
    entries = [check_worker_entry(raw, source=WORKER_ENV) for raw in _load(WORKER_ENV)]
    repeated = _duplicates([entry['name'] for entry in entries])
    if repeated:
        raise MCPConfigError(f'{WORKER_ENV}: servers declared more than once: {repeated}')
    return entries


__all__ = [
    'DEFAULT_RUNTIME',
    'USER_ENV',
    'WORKER_ENV',
    'WORKER_KEYS',
    'MCPConfigError',
    'check_worker_entry',
    'normalise_user_entry',
    'user_servers',
    'worker_servers',
]
