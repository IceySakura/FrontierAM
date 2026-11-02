"""Host side of the Codex print backend: drive a turn, read its event stream.

``send_turn`` runs one ``codex exec`` in the sandbox and turns its JSONL stdout
into the turn dict the manager reads. The stream's shape is documented in
``agents/codex/json_events.py``, whose parser this reuses.
"""

import logging

from collections.abc import Mapping, Sequence
from typing import Any

from nexus.extensions.agents.codex.json_events import parse_codex_jsonl
from nexus.extensions.tasks.hitl.actor.memory import place_user_memory
from nexus.extensions.tasks.hitl.actor.print_codex import runtime as sandbox
from nexus.extensions.tasks.hitl.actor.print_codex.rendering import TOOL_ITEM_TYPES, item_detail
from nexus.extensions.tasks.hitl.actor.print_codex.spec import mcp_overrides
from nexus.extensions.tasks.hitl.actor.skills import place_worker_skills
from nexus.runtime import Runtime
from nexus.runtime.models import InvokeRequest


logger = logging.getLogger(__name__)


DEFAULT_TURN_TIMEOUT_SECONDS = 7200

# Event types carried by ``codex exec --json``.
THREAD_STARTED = 'thread.started'
TURN_FAILED = 'turn.failed'
ITEM_COMPLETED = 'item.completed'
ERROR = 'error'


class CodexPrintBackend:
    """One Codex install, one CODEX_HOME and one labelled proxy per container."""

    def __init__(self, config: Mapping[str, Any], *, memory_path: str = '', skills_path: str = '') -> None:
        self._config = dict(config)
        self._memory_path = memory_path
        self._skills_path = skills_path
        self._env_ready = False
        self._open_sessions: set[str] = set()

    @property
    def open_sessions(self) -> frozenset[str]:
        return frozenset(self._open_sessions)

    async def ensure_env(self, runtime: Runtime) -> None:
        """Install the Codex CLI and place what it reads at startup. Once per container."""
        if self._env_ready:
            return
        await runtime.invoke(
            sandbox.setup_env,
            version=self._version,
            npm_registry=self._config.get('npm_registry'),
        )
        await place_user_memory(runtime, self._config, memory_path=self._memory_path)
        await place_worker_skills(runtime, self._config, skills_path=self._skills_path)
        self._env_ready = True

    @property
    def _version(self) -> str:
        """The Codex npm version this run installs and writes its catalog for."""
        return str(self._config.get('version') or 'latest')

    async def open_session(self, runtime: Runtime, *, session_id: str, label: str) -> int:
        """Start this session's labelled proxy and write its config. Returns its port."""
        await self.ensure_env(runtime)
        result = await runtime.invoke(
            sandbox.open_session,
            session_id=session_id,
            label=label,
            config=self._config,
        )
        self._open_sessions.add(session_id)
        port = int((result or {}).get('proxy_port') or 0)
        logger.info('codex session %s open on proxy port %d', session_id, port)
        return port

    async def send_turn(
        self,
        runtime: Runtime,
        *,
        session_id: str,
        prompt: str,
        thread_id: str,
        working_dir: str,
        timeout: float,
        extra_env: dict[str, str] | None = None,
        extra_cli_args: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run one turn and interpret its event stream."""
        if session_id not in self._open_sessions:
            raise RuntimeError(f'session {session_id} was never opened; call open_session() first')

        # ``disable_invoke_wait_timeout``: the sandbox runs the turn under
        # ``timeout``, which is a whole agent loop rather than one RPC.
        raw = await runtime.invoke(
            InvokeRequest.build(
                sandbox.run_turn,
                session_id=session_id,
                prompt=prompt,
                thread_id=thread_id,
                working_dir=working_dir,
                timeout=timeout,
                extra_env=extra_env,
                extra_cli_args=[*self._mcp_overrides(), *(extra_cli_args or [])],
                venv_id=runtime.default_venv_id,
                disable_invoke_wait_timeout=True,
            )
        )
        return interpret_turn(raw, session_id=session_id)

    def _mcp_overrides(self) -> list[str]:
        """The MCP registrations every turn carries, resumed turns included."""
        servers = self._config.get('mcp_servers')
        if not isinstance(servers, Sequence) or isinstance(servers, (str, bytes)):
            return []
        return mcp_overrides([server for server in servers if isinstance(server, Mapping)])

    async def close_session(self, runtime: Runtime, *, session_id: str) -> None:
        if session_id not in self._open_sessions:
            return
        try:
            await runtime.invoke(sandbox.close_session, session_id=session_id)
        except Exception as exc:  # noqa: BLE001 - teardown must not mask the real error
            logger.warning('failed to close codex session %s: %s', session_id, exc)
        self._open_sessions.discard(session_id)

    async def shutdown(self, runtime: Runtime) -> None:
        """Stop every proxy. Safe to call when setup never happened."""
        if self._env_ready:
            try:
                await runtime.invoke(sandbox.teardown_env)
            except Exception as exc:  # noqa: BLE001
                logger.warning('codex teardown failed: %s', exc)
            self._env_ready = False
        self._open_sessions.clear()


def interpret_turn(raw: Any, *, session_id: str) -> dict[str, Any]:
    """The sandbox's raw result as the manager's turn dict."""
    raw_dict = raw if isinstance(raw, dict) else {}
    stdout = str(raw_dict.get('output', ''))

    exit_code_value = raw_dict.get('exit_code')
    exit_code = int(exit_code_value) if exit_code_value is not None else 0

    events = parse_codex_jsonl(stdout)
    return {
        'success': exit_code == 0 and not _failed(events),
        'session_id': session_id,
        'thread_id': thread_id_of(events),
        'output': final_message(events),
        'steps': steps_of(events),
        'failure': _failure_message(events),
        'stderr': str(raw_dict.get('stderr', '')),
        'exit_code': exit_code,
        # The command was killed at its timeout, which the manager ends the session on.
        'timed_out': bool(raw_dict.get('timed_out')),
        'execution_time': raw_dict.get('execution_time', 0),
    }


def thread_id_of(events: Sequence[dict[str, Any]]) -> str:
    """The thread this turn ran under, from ``thread.started``.

    Emitted on a resumed thread as well as a new one, carrying the same id, so a
    caller can read it back every turn.
    """
    for event in events:
        if event.get('type') == THREAD_STARTED and (found := str(event.get('thread_id') or '')):
            return found
    return ''


def steps_of(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """What the worker said and the items it completed, in order.

    Shaped as the ``text`` / ``tool_use`` entries ``actor.rendering.read_steps``
    reads, each tool step carrying the ``detail`` picked out here.
    """
    steps: list[dict[str, Any]] = []
    for event in events:
        if event.get('type') != ITEM_COMPLETED:
            continue
        item = event.get('item')
        if not isinstance(item, Mapping):
            continue
        item_type = str(item.get('item_type') or item.get('type') or '')
        if item_type == 'agent_message':
            if text := str(item.get('text') or ''):
                steps.append({'type': 'text', 'text': text})
        elif item_type in TOOL_ITEM_TYPES:
            steps.append({'type': 'tool_use', 'name': item_type, 'detail': item_detail(item_type, item)})
    return steps


def final_message(events: Sequence[dict[str, Any]]) -> str:
    """The last thing the worker said, which is its claim about the turn."""
    said = [step['text'] for step in steps_of(events) if step.get('type') == 'text']
    return said[-1] if said else ''


def _failed(events: Sequence[dict[str, Any]]) -> bool:
    return any(event.get('type') in {TURN_FAILED, ERROR} for event in events)


def _failure_message(events: Sequence[dict[str, Any]]) -> str:
    """What the last failure event said, or '' when the stream names none."""
    for event in reversed(events):
        event_type = event.get('type')
        if event_type == TURN_FAILED:
            error = event.get('error')
            return str(error.get('message') or '') if isinstance(error, Mapping) else ''
        if event_type == ERROR:
            return str(event.get('message') or '')
    return ''


__all__ = [
    'DEFAULT_TURN_TIMEOUT_SECONDS',
    'TOOL_ITEM_TYPES',
    'CodexPrintBackend',
    'final_message',
    'interpret_turn',
    'steps_of',
    'thread_id_of',
]
