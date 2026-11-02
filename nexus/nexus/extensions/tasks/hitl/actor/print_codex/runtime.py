"""Sandbox-side functions for the labelled Codex print backend.

Invoked through ``runtime.invoke()``, so this runs inside the sandbox process and
its module-level state persists across calls within one container.

Lifecycle::

    1. setup_env()               install the Codex CLI -- once per container
    2. open_session()            one labelled proxy, against the container's CODEX_HOME
    3. run_turn()      * N       one ``codex exec`` process per turn
    4. close_session() / teardown_env()

The session holds one label for its proxy's lifetime, which is what attributes its
turns. ``CODEX_HOME`` is one fixed directory for the container, the way a person's own
``~/.codex`` is: the ``config.toml``, the ``AGENTS.md`` and the catalog there hold what
every session shares, and Codex keys the threads it persists by id. The one value that
differs per session is its proxy's port, which ``run_turn`` passes as
``-c model_providers.<id>.base_url`` rather than writing to that shared file.
"""

import contextlib
import json
import logging
import os
import shlex
import time

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

_codex_entry: str | None = None
_node_path: str | None = None

# What the container was already configured with, so a second call carrying
# different settings is reported rather than skipped.
_setup_signature: str | None = None

# session_id -> its own proxy, port, label and CODEX_HOME.
_sessions: dict[str, '_CodexSession'] = {}

_DEFAULT_INSTALL_DIR = '/tmp/codex-cli'
_DEFAULT_TURN_TIMEOUT_SECONDS = 7200

# What a turn reports when the command was killed at its timeout, which leaves no
# exit code of its own. 124 is what ``timeout(1)`` uses.
_TIMEOUT_EXIT_CODE = 124

# Where Codex keeps its config, its instructions and the threads ``resume`` reads
# back. Fixed, and shared by every session in the container: ``spec.MEMORY_PATH``
# names ``.codex/AGENTS.md`` literally, and a directory per session left the worker
# reading no AGENTS.md at all with nothing raised.
_CODEX_HOME = '.codex'

# The provider id in the container's ``config.toml``: the ``model_provider`` value,
# the ``[model_providers.<id>]`` table name, and nothing outside that file.
_PROVIDER_ID = 'proxy'

# The provider's display name when ``use_azure_provider`` is False.
_PROVIDER_LABEL = 'Local Proxy'

# Names the ``env_key`` in config.toml and the variable ``run_turn`` sets.
_PROVIDER_KEY_ENV = 'CODEX_PROXY_API_KEY'


def _proxy_base_url(port: int) -> str:
    """Where a session's own proxy listens."""
    return f'http://127.0.0.1:{port}/v1'


@dataclass
class _CodexSession:
    """One session's proxy, its port, the label it stamps, and its Codex home."""

    proxy: Any
    port: int
    label: str
    codex_home: str


async def setup_env(
    *,
    version: str = 'latest',
    npm_registry: str | None = None,
) -> dict[str, Any]:
    """Install the Codex CLI. Idempotent, once per container.

    Raises when called again with different settings: the install below would be
    skipped and the caller would run against the first call's version.
    """
    global _codex_entry, _node_path, _setup_signature

    from nexus.extensions.agents.codex.installer import install_codex_cli
    from nexus.extensions.agents.installed_common import (
        find_nexus_runtime_binary,
        resolve_npm_bin_entry,
    )

    signature = repr((version, npm_registry))
    if _setup_signature is not None:
        if signature != _setup_signature:
            raise RuntimeError(
                f'Codex is already set up in this container with {_setup_signature}, '
                f'and this call asks for {signature}. The install is skipped when it is '
                'already present, so the second set of settings would not take effect.'
            )
        return {'codex_entry': _codex_entry, 'node_path': _node_path}

    await install_codex_cli(install_dir=_DEFAULT_INSTALL_DIR, version=version, npm_registry=npm_registry)

    node_path = find_nexus_runtime_binary('node')
    if not node_path:
        raise RuntimeError('Node.js is required but not found in the runtime')
    entry = resolve_npm_bin_entry(_DEFAULT_INSTALL_DIR, '@openai/codex', 'codex')
    if not entry or not entry.exists():
        raise RuntimeError(f'Codex CLI JS entry point not found under {_DEFAULT_INSTALL_DIR}')

    _node_path = node_path
    _codex_entry = str(entry)
    _setup_signature = signature
    logger.info('Codex CLI ready: %s', _codex_entry)
    return {'codex_entry': _codex_entry, 'node_path': _node_path}


async def open_session(
    *,
    session_id: str,
    label: str,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Start this session's own labelled proxy and write the container's ``config.toml``.

    The label is fixed for the proxy's lifetime, which is what makes concurrent
    turns safe: there is no per-turn slot for another session to overwrite.

    Opening a second session rewrites the shared ``config.toml`` with the same values
    apart from ``base_url``, which ``run_turn`` overrides per turn anyway.

    ``config`` is the run's whole worker config, the mapping the manager holds.
    """
    from nexus.extensions.agents.installed_common import find_free_port
    from nexus.extensions.tasks.hitl.actor.labelling.proxy import attach_label
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.server.services.responses_proxy import ResponsesProxy

    if not session_id:
        raise ValueError('session_id must be non-empty')
    if not label:
        raise ValueError('label must be non-empty; an unlabelled session cannot be attributed')

    cfg: Mapping[str, Any] = config or {}
    version = str(cfg.get('version') or 'latest')

    existing = _sessions.get(session_id)
    if existing is not None:
        if existing.label != label:
            raise ValueError(f'session {session_id} is already open with label {existing.label!r}, not {label!r}')
        return {'proxy_port': existing.port, 'label': existing.label, 'codex_home': existing.codex_home}

    codex_home = Path(os.environ.get('HOME', '/root')) / _CODEX_HOME
    codex_home.mkdir(parents=True, exist_ok=True)

    rt = require_runtime_control()
    port = find_free_port()
    proxy = ResponsesProxy(
        sio=rt.sio,
        listen_port=port,
        listen_host='127.0.0.1',
        # The command owns the turn's wall clock, so a second deadline here would
        # expire while the host is still serving the call.
        request_timeout=None,
        codex_version=version,
        **_converter_kwargs(cfg),
    )
    await proxy.start()
    attach_label(proxy, label)

    catalog_path = _write_catalog(codex_home, catalog=cfg.get('model_catalog_json'))

    _write_config(codex_home, proxy_port=port, config=cfg, model_catalog_path=catalog_path)

    _sessions[session_id] = _CodexSession(proxy=proxy, port=port, label=label, codex_home=str(codex_home))
    logger.info('Session %s proxy on port %d labelling as %s, home %s', session_id, port, label, codex_home)
    return {'proxy_port': port, 'label': label, 'codex_home': str(codex_home)}


def _converter_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    """The converter this session's proxy runs, when the run asks for a different one.

    ``wrap_plaintext_reasoning_content`` False stops the converter minting an ``rs_*``
    for a plaintext ``reasoning_content``. Codex replays that id on the next turn and
    a stateless upstream answers ``rs_ not found``.
    """
    if config.get('wrap_plaintext_reasoning_content', True):
        return {}

    from functools import partial

    from nexus.runtime.server.services.responses_proxy.converter import openai_to_responses_events

    return {
        'openai_to_responses_events_converter': partial(
            openai_to_responses_events,
            wrap_plaintext_reasoning_content=False,
        )
    }


def _write_catalog(codex_home: Path, *, catalog: Any) -> Path | None:
    """Materialize this session's model catalog. Returns its path, or None.

    None leaves the catalog Codex bundles in effect. The file is written by
    :func:`nexus.extensions.agents.codex.runtime_setup._write_model_catalog_json`,
    shared with the one-shot Codex agent.
    """
    if catalog is None:
        return None
    if not isinstance(catalog, dict):
        raise TypeError(f'model_catalog_json must be a mapping, got {type(catalog).__name__}')

    from nexus.extensions.agents.codex.runtime_setup import _write_model_catalog_json

    return _write_model_catalog_json(codex_home, catalog)


def _write_config(
    codex_home: Path,
    *,
    proxy_port: int,
    config: Mapping[str, Any],
    model_catalog_path: Path | None = None,
) -> None:
    """Point Codex at this session's proxy, in its own ``config.toml``.

    Every top-level key is written before the first table header, since a key after
    one belongs to that table.

    The provider name "azure" keeps ``store=true``, re-attaches the original
    reasoning item ids on replay, and routes compaction to ``/responses/compact``.

    ``user_input_tools`` False writes the ``[tools]`` entry that keeps
    ``request_user_input`` off the wire.

    The ``base_url`` written here is whichever session opened last; ``run_turn``
    overrides it per turn with the port of the session that turn belongs to.
    """
    lines: list[str] = []
    if model := config.get('codex_model'):
        lines.append(f'model = {json.dumps(str(model))}')
    if model_catalog_path is not None:
        lines.append(f'model_catalog_json = {json.dumps(str(model_catalog_path))}')
    # Codex compacts at min(limit, window * 90%).
    if (window := config.get('model_context_window')) is not None:
        lines.append(f'model_context_window = {int(window)}')
    if (limit := config.get('auto_compact_token_limit')) is not None:
        lines.append(f'model_auto_compact_token_limit = {int(limit)}')
    # Absent, the catalog entry's ``default_reasoning_level`` applies.
    if effort := config.get('reasoning_effort'):
        lines.append(f'model_reasoning_effort = {json.dumps(str(effort))}')
    # The proxy serves no search backend.
    lines.append('web_search = "disabled"')
    lines.extend([
        f'model_provider = {json.dumps(_PROVIDER_ID)}',
        '',
        '[features]',
        # Compaction stays local; the remote path posts to /responses/compact.
        'remote_compaction_v2 = false',
    ])
    if not config.get('user_input_tools'):
        lines.extend([
            '',
            # Codex answers a call to ``request_user_input`` itself with
            # ``unavailable in Default mode``, since no one is at a terminal to answer it.
            '[tools.experimental_request_user_input]',
            'enabled = false',
        ])
    lines.extend([
        '',
        f'[model_providers.{_PROVIDER_ID}]',
        f'name = {json.dumps("azure" if config.get("use_azure_provider") else _PROVIDER_LABEL)}',
        f'base_url = {json.dumps(_proxy_base_url(proxy_port))}',
        f'env_key = {json.dumps(_PROVIDER_KEY_ENV)}',
        'wire_api = "responses"',
    ])
    (codex_home / 'config.toml').write_text('\n'.join(lines) + '\n')


async def run_turn(
    *,
    session_id: str,
    prompt: str,
    thread_id: str = '',
    working_dir: str = '/workspace',
    timeout: float = _DEFAULT_TURN_TIMEOUT_SECONDS,
    extra_env: dict[str, str] | None = None,
    extra_cli_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run one ``codex exec`` turn. Returns when the process exits.

    ``thread_id`` resumes the thread Codex persisted under this session's
    ``CODEX_HOME``; empty starts a new one. Reads only immutable state, so
    concurrent turns for different sessions do not interfere.
    """
    from nexus.extensions.agents.installed_common import DEFAULT_PROXY_API_KEY, build_runtime_path
    from nexus.runtime.executors.command import RunCommandRequest, run_command

    if not _codex_entry or not _node_path:
        raise RuntimeError('Codex is not installed. Call setup_env() first.')
    session = _sessions.get(session_id)
    if session is None:
        raise RuntimeError(f'Session {session_id} has no proxy. Call open_session() first.')

    start_time = time.time()

    if not Path(working_dir).is_dir():
        logger.warning('working_dir %s does not exist in the sandbox; creating it empty', working_dir)
        Path(working_dir).mkdir(parents=True, exist_ok=True)

    codex_env = {
        'CODEX_HOME': session.codex_home,
        _PROVIDER_KEY_ENV: DEFAULT_PROXY_API_KEY,
        'HOME': os.environ.get('HOME', '/root'),
        'PATH': build_runtime_path(),
    }
    if extra_env:
        codex_env.update(extra_env)

    cmd: list[str] = [
        _node_path,
        _codex_entry,
        'exec',
        '--json',
        '--dangerously-bypass-approvals-and-sandbox',
        '--skip-git-repo-check',
        # The port is this session's, while CODEX_HOME holds one config.toml for the
        # container. Passing it per turn is what lets several sessions share that file.
        '-c',
        f'model_providers.{_PROVIDER_ID}.base_url={json.dumps(_proxy_base_url(session.port))}',
        '-C',
        working_dir,
    ]
    if extra_cli_args:
        cmd.extend(extra_cli_args)
    # ``resume <id>`` is a subcommand of ``exec``; the flags above belong before it
    # and the prompt after it.
    if thread_id:
        cmd.extend(['resume', thread_id])
    cmd.append(prompt)

    shell_cmd = ' '.join(shlex.quote(part) for part in cmd)
    logger.info('Running codex turn (%s, thread=%s): %s', session_id, thread_id or '(new)', shell_cmd[:300])

    result = await run_command(
        RunCommandRequest(
            command=shell_cmd,
            cwd=working_dir,
            timeout=timeout,
            extra_env=codex_env,
        )
    )

    stdout = result.stdout or ''
    raw_exit_code = result.return_code
    # ``run_command`` leaves no exit code when it kills the process at ``timeout``.
    timed_out = raw_exit_code is None
    if timed_out:
        logger.warning('codex turn produced no exit code (session_id=%s) after %.0fs', session_id, timeout)
    exit_code = _TIMEOUT_EXIT_CODE if timed_out else raw_exit_code
    execution_time = time.time() - start_time

    logger.info(
        'codex turn done (%s): exit_code=%s time=%.1fs output=%d chars',
        session_id,
        exit_code,
        execution_time,
        len(stdout),
    )

    return {
        'output': stdout,
        'stderr': result.stderr or '',
        'exit_code': exit_code,
        'timed_out': timed_out,
        'execution_time': execution_time,
        'session_id': session_id,
    }


async def close_session(*, session_id: str) -> None:
    """Stop one session's proxy, freeing its port. Leaves its CODEX_HOME in place."""
    session = _sessions.pop(session_id, None)
    if session is None:
        return
    with contextlib.suppress(Exception):
        await session.proxy.stop()
    logger.info('Session %s proxy on port %d stopped', session_id, session.port)


async def teardown_env() -> None:
    """Stop every remaining proxy."""
    for session_id in list(_sessions):
        await close_session(session_id=session_id)


__all__ = [
    'close_session',
    'open_session',
    'run_turn',
    'setup_env',
    'teardown_env',
]
