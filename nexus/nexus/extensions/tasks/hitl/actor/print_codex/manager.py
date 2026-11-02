"""Codex print-mode session manager.

One ``codex exec`` process per turn, resumed by the thread id Codex reports. Each
session owns its own sandbox proxy, which fixes its trace label and so lets
sessions run concurrently -- ``SessionManager.run_turns`` drives that.

``Session.session_id`` is empty until the first turn comes back: Codex mints the
thread id itself and reports it on ``thread.started``, which the host reads off
every turn's event stream.
"""

import asyncio
import logging
import time

from collections.abc import Mapping
from dataclasses import dataclass, field

from nexus.core import require_context
from nexus.extensions.tasks.hitl.actor import envs
from nexus.extensions.tasks.hitl.actor.base import (
    STATUS_UNREACHABLE,
    Session,
    SessionManager,
    SessionMessage,
    TurnKind,
    TurnResult,
)
from nexus.extensions.tasks.hitl.actor.cost import account_worker_turn, append_cost
from nexus.extensions.tasks.hitl.actor.input_spec import (
    BUILTIN_COMMANDS_KEY,
    WORKER_COMMANDS_KEY,
    check_outgoing,
    offered_builtins,
)
from nexus.extensions.tasks.hitl.actor.labelling.handler import LabelledLLMRequestHandler
from nexus.extensions.tasks.hitl.actor.print_codex.backend import (
    DEFAULT_TURN_TIMEOUT_SECONDS,
    CodexPrintBackend,
)
from nexus.extensions.tasks.hitl.actor.print_codex.spec import (
    BUILTIN_COMMANDS,
    COMMAND_PREFIX,
    DENIED_COMMANDS,
    MEMORY_PATH,
    SKILLS_PATH,
    UNSUPPORTED_COMMANDS,
    worker_env,
)
from nexus.extensions.tasks.hitl.actor.registry import worker_manager_registry
from nexus.extensions.tasks.hitl.actor.rendering import DEFAULT_DETAIL, Detail, report_from_steps
from nexus.extensions.tasks.hitl.actor.resources import acquire_llm_handler, release_llm_handler
from nexus.extensions.tasks.hitl.trace import labels
from nexus.extensions.tasks.hitl.utils import coerce
from nexus.extensions.tasks.hitl.utils.text import tagged
from nexus.runtime import Runtime


logger = logging.getLogger(__name__)


NO_THREAD_REASON = (
    'The worker did not report a thread to continue from, so the next message would '
    'reach a fresh Codex with no memory of this one.'
)
"""Ends a run whose turn produced no thread id."""

TIMED_OUT_REASON = (
    'The worker spent its whole turn budget without finishing, so the command was '
    'killed. A resumed thread would stall the same way.'
)
"""Ends a run whose turn was killed at its timeout."""

NO_MESSAGES_NOTE = 'This turn produced no separable messages; only its combined output is available.'

FAILED_TURN_NOTE = 'The worker stopped on an error: {reason}'
"""Carries what the event stream named. The thread survives, so the next message
still reaches the same conversation."""


@dataclass
class PrintCodexSession(Session):
    """One resumable Codex thread.

    ``name`` identifies it host-side and carries the trace label; ``session_id`` is
    the thread id Codex reports, empty until the first turn.
    """

    workspace: str = '/workspace'
    config: dict[str, object] = field(default_factory=dict)
    turns: int = 0

    @property
    def label(self) -> str:
        """The trace label, keyed on ``name`` rather than the thread id."""
        return labels.worker_label(self.name)


@worker_manager_registry.register('print_codex')
class PrintCodexSessionManager(SessionManager):
    """Drive many ``codex exec`` threads, concurrently if asked."""

    backend = 'print_codex'
    product = 'Codex'
    memory_path = MEMORY_PATH
    skills_path = SKILLS_PATH
    command_prefix = COMMAND_PREFIX
    denied_commands = DENIED_COMMANDS
    unsupported_commands = UNSUPPORTED_COMMANDS

    def __init__(
        self,
        *,
        default_working_dir: str = '/workspace',
        default_config: Mapping[str, object] | None = None,
        openai_client_name: str = 'main',
        runtime_name: str = 'main',
        llm_handler: LabelledLLMRequestHandler | None = None,
    ) -> None:
        self._default_working_dir = default_working_dir
        self._default_config = dict(default_config or {})
        self._openai_client_name = openai_client_name
        self._runtime_name = runtime_name
        self._sessions_by_name: dict[str, PrintCodexSession] = {}
        self._start_lock = asyncio.Lock()
        self._runtime: Runtime | None = None
        self._handler = llm_handler
        self._owns_handler = False
        self._backend = CodexPrintBackend(
            self._default_config, memory_path=self.memory_path, skills_path=self.skills_path
        )

    @property
    def builtin_commands(self) -> Mapping[str, str]:
        """The harness's own commands this run offers, from ``worker_config``."""
        return offered_builtins(
            BUILTIN_COMMANDS,
            coerce.optional_str_list(self._default_config.get(BUILTIN_COMMANDS_KEY)) or (),
        )

    @property
    def run_commands(self) -> Mapping[str, str]:
        """The skills this run installed, from ``worker_config['worker_commands']``."""
        return coerce.string_map(self._default_config.get(WORKER_COMMANDS_KEY))

    def harness_env(self, config: Mapping[str, object]) -> dict[str, str]:
        return worker_env(config)

    # -- lifecycle ---------------------------------------------------------

    async def setup(self) -> None:
        if self._runtime is not None:
            return

        ctx = require_context()
        runtime = await ctx.get_runtime(self._runtime_name)
        for attempt in range(3):
            try:
                await runtime.ws_connect()
                break
            except Exception as exc:  # noqa: BLE001 - retried below, re-raised on the last attempt
                if attempt == 2:
                    raise
                wait = 5.0 * (2**attempt)
                logger.warning('ws_connect attempt %d failed: %s; retrying in %.0fs', attempt + 1, exc, wait)
                await asyncio.sleep(wait)

        self._runtime = runtime
        if self._handler is None:
            self._handler = acquire_llm_handler(
                runtime,
                openai_client=ctx.get_openai_client(self._openai_client_name),
                no_thinking_request_args=coerce.optional_dict(self._default_config.get('no_thinking_request_args')),
                haiku_model=coerce.optional_string(self._default_config.get('haiku_model')),
                preserve_thinking=coerce.optional_bool(self._default_config.get('preserve_thinking'), default=True),
                factory=LabelledLLMRequestHandler,
            )
            # Only the manager that acquired the handler may release it.
            self._owns_handler = True

    async def start(
        self,
        *,
        name: str,
        workspace: str = '',
        metadata: Mapping[str, object] | None = None,
        config: Mapping[str, object] | None = None,
        **kwargs: object,
    ) -> PrintCodexSession:
        """Open a session and its sandbox proxy.

        The proxy opens here rather than on the first turn, so a bad label or a
        sandbox problem surfaces at session start. ``session_id`` stays empty until
        the first turn reports its thread.
        """
        if self._runtime is None:
            raise RuntimeError('PrintCodexSessionManager is not set up. Call setup() first.')
        # Installation and shared config writes finish before another session opens.
        async with self._start_lock:
            labels.validate_name(name, kind='session name')
            if name in self._sessions_by_name:
                raise ValueError(f'print Codex session already exists: {name}')

            session = PrintCodexSession(
                name=name,
                session_id='',
                backend=self.backend,
                metadata=dict(metadata or {}),
                workspace=workspace or self._default_working_dir,
                config={**self._default_config, **dict(config or {})},
            )

            await self._backend.open_session(self._runtime, session_id=name, label=session.label)

            self._sessions_by_name[name] = session
            return session

    def get(self, name: str) -> PrintCodexSession:
        return self._sessions_by_name[name]

    def session_of(self, session_id: str) -> PrintCodexSession | None:
        """The session holding this thread id, or None before its first turn."""
        if not session_id:
            return None
        return next(
            (session for session in self._sessions_by_name.values() if session.session_id == session_id),
            None,
        )

    # -- turns -------------------------------------------------------------

    async def send(self, session: Session, message: SessionMessage) -> TurnResult:
        """Run one turn. Safe to call concurrently for different sessions."""
        if self._runtime is None:
            raise RuntimeError('PrintCodexSessionManager is not set up. Call setup() first.')

        codex_session = _require_codex_session(session)

        verdict = check_outgoing(
            message.content,
            allowed=self.commands,
            prefix=self.command_prefix,
            denied=self.denied_commands,
            unsupported=self.unsupported_commands,
        )
        if not verdict.allowed:
            logger.warning('[codex %s] blocked outgoing message: %s', codex_session.name, verdict.reason)
            return TurnResult(
                kind='refused',
                session_id=codex_session.session_id,
                raw={'status': 'blocked_by_guard', 'reason': verdict.reason},
                metadata={
                    'message_kind': message.kind,
                    'name': codex_session.name,
                    'blocked': True,
                    'reason': verdict.reason,
                },
            )

        turn = codex_session.turns + 1
        started = time.monotonic()
        try:
            result = await self._backend.send_turn(
                self._runtime,
                session_id=codex_session.name,
                prompt=message.content,
                thread_id=codex_session.session_id,
                working_dir=codex_session.workspace,
                timeout=_turn_timeout(codex_session.config),
                extra_env=envs.session_env(
                    codex_session.config,
                    session=codex_session,
                    harness_env=self.harness_env(codex_session.config),
                ),
                extra_cli_args=coerce.optional_str_list(codex_session.config.get('extra_cli_args')),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported as a lost session, not raised
            codex_session.turns = turn
            logger.warning('[codex %s] turn %d could not be run: %s', codex_session.name, turn, exc)
            return TurnResult(
                kind='terminal',
                session_id=codex_session.session_id,
                raw={'status': STATUS_UNREACHABLE, 'reason': f'{type(exc).__name__}: {exc}'},
                metadata={
                    'message_kind': message.kind,
                    'name': codex_session.name,
                    'turn': turn,
                    'status': STATUS_UNREACHABLE,
                    'error': f'{type(exc).__name__}: {exc}',
                },
            )

        codex_session.turns = turn
        elapsed = time.monotonic() - started
        raw = {str(key): value for key, value in result.items()}

        # Reported on a resumed thread as well as a new one, carrying the same id.
        reported = str(raw.get('thread_id') or '')
        if reported:
            if codex_session.session_id and reported != codex_session.session_id:
                logger.warning(
                    '[codex %s] turn %d reported thread %s, not %s; continuing from the new one',
                    codex_session.name,
                    turn,
                    reported,
                    codex_session.session_id,
                )
            codex_session.session_id = reported
        else:
            logger.warning('[codex %s] turn %d reported no thread id', codex_session.name, turn)
            raw.setdefault('reason', NO_THREAD_REASON)

        timed_out = bool(raw.get('timed_out'))
        if timed_out:
            logger.warning(
                '[codex %s] turn %d was killed at its timeout after %.0fs; ending the session',
                codex_session.name,
                turn,
                elapsed,
            )
            raw.setdefault('reason', TIMED_OUT_REASON)

        return TurnResult(
            kind=_turn_kind(thread_id=codex_session.session_id, timed_out=timed_out),
            output=str(raw.get('output') or ''),
            session_id=codex_session.session_id,
            raw=raw,
            cost=account_worker_turn(codex_session, seconds=elapsed),
            metadata={
                'message_kind': message.kind,
                'name': codex_session.name,
                'turn': turn,
                'label': codex_session.label,
            },
        )

    # -- reporting ---------------------------------------------------------

    def report(self, result: TurnResult, *, detail: Detail = DEFAULT_DETAIL) -> str:
        """Project the turn from the steps this backend records.

        A failure the stream named is appended as its own section, whether or not
        the turn also produced messages.
        """
        rendered = report_from_steps(
            result.raw.get('steps'),
            output=result.output,
            question=result.question,
            empty_note=NO_MESSAGES_NOTE,
            detail=detail,
        )
        failure = str(result.raw.get('failure') or '')
        if failure:
            rendered = f'{rendered}\n\n{tagged("note", FAILED_TURN_NOTE.format(reason=failure))}'
        return append_cost(rendered, result.cost)

    # -- traces ------------------------------------------------------------

    def traces(self, session: Session) -> dict[str, object]:
        """This session's own trajectories, from the run-level tracker."""
        return labels.filter_by_label(labels.run_level_trajectories(), session.label)

    def unattributed_requests(self) -> int:
        """Requests that arrived with no resolvable label. Should be zero."""
        if self._handler is None:
            return 0
        return self._handler.unattributed_requests

    # -- teardown ----------------------------------------------------------

    async def close(self, session: Session) -> None:
        codex_session = _require_codex_session(session)
        if self._runtime is not None:
            await self._backend.close_session(self._runtime, session_id=codex_session.name)
        self._sessions_by_name.pop(codex_session.name, None)

    async def shutdown(self) -> None:
        if self._runtime is not None:
            try:
                await self._backend.shutdown(self._runtime)
            except Exception as exc:  # noqa: BLE001 - shutdown must not mask the outcome
                logger.warning('codex backend shutdown failed: %s', exc)
            if self._owns_handler:
                release_llm_handler(self._runtime)
                self._owns_handler = False
        self._sessions_by_name.clear()
        self._runtime = None


def _turn_kind(*, thread_id: str, timed_out: bool) -> TurnKind:
    """How the turn ended.

    Terminal where there is no thread to continue from, which is what a next send
    would need: a failed turn leaves the thread on disk, so ``resume`` still
    reaches the same conversation.

    Terminal too when the command was killed at its timeout. The turn spent its
    whole budget and reported nothing, and a resumed thread hits the same wall:
    what stalls it is the session's own proxy, which lives as long as the session.
    """
    if timed_out:
        return 'terminal'
    return 'completed' if thread_id else 'terminal'


def _turn_timeout(config: Mapping[str, object]) -> float:
    return float(coerce.optional_int(config.get('timeout')) or DEFAULT_TURN_TIMEOUT_SECONDS)


def _require_codex_session(session: Session) -> PrintCodexSession:
    if not isinstance(session, PrintCodexSession):
        raise TypeError(f'expected a PrintCodexSession, got {type(session).__name__}')
    return session


__all__ = [
    'NO_THREAD_REASON',
    'TIMED_OUT_REASON',
    'PrintCodexSession',
    'PrintCodexSessionManager',
]
