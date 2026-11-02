"""Interactive PTY Claude Code worker backend.

Where the ``claude_code`` backend drives ``claude --print`` over pipes, this one
drives the real interactive TUI running inside a PTY in the sandbox. That
exercises Claude Code's interactive-only branch (different tool set, different
system prompt, interactive-only features) and lets the worker *ask the host
questions* mid-turn through its hooks.

The host sees one input operation: :meth:`PtyClaudeCodeSessionManager.send`. When
the worker is blocked on a hook, the next ``send`` is routed internally as the
answer to it — the caller does not switch APIs, and does not name the pending
interaction. Callers decide what a turn means by inspecting
``TurnResult.user_input_request``:

* ``source == 'hook'`` — the worker is blocked and *must* be answered next.
* otherwise ``success`` — an ordinary turn; the caller is free to do anything.
* otherwise — terminal (see :func:`runtime_adapter.session_died`).

``user_input_request`` is left ``None`` on ordinary turns precisely so the first
check cannot be approximated by a presence test.
"""

import asyncio
import logging
import time
import uuid

from collections.abc import Mapping
from dataclasses import dataclass, field

from nexus.core import UserInputRequest, require_context
from nexus.extensions.agents.installed_common.multiturn import normalize_stream_json_session_id
from nexus.extensions.tasks.hitl.actor import envs
from nexus.extensions.tasks.hitl.actor.base import (
    STATUS_ANSWER_MISSED,
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
from nexus.extensions.tasks.hitl.actor.memory import place_user_memory
from nexus.extensions.tasks.hitl.actor.print_claude_code.rendering import steps_with_detail
from nexus.extensions.tasks.hitl.actor.pty_claude_code import runtime_adapter
from nexus.extensions.tasks.hitl.actor.pty_claude_code.spec import (
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
from nexus.extensions.tasks.hitl.actor.skills import place_worker_skills
from nexus.extensions.tasks.hitl.trace import labels
from nexus.extensions.tasks.hitl.utils import coerce
from nexus.runtime import Runtime


logger = logging.getLogger(__name__)


ANSWER_MISSED_REASON = (
    'For technical reasons your answer did not reach them, and they picked an answer '
    'themselves and carried on. Ask them what they decided, or say what you wanted '
    'instead -- they can still change it.'
)
"""What the user is told. States the outcome and what to do about it, not the cause.

The user has no way to act on a timeout and no reason to know the conversation is
mediated, so naming the mechanism would only tell them they are inside a harness.
"""


@dataclass
class PtyClaudeCodeSession(Session):
    """One long-lived interactive PTY conversation."""

    workspace: str = '/workspace'
    config: dict[str, object] = field(default_factory=dict)
    # Set while the worker is blocked on a hook. Its presence is what routes the
    # next send to ``respond`` instead of starting a new turn.
    pending_interaction_id: str | None = None
    # The questions that hook is waiting on. Their count decides the shape the
    # answer has to take; see ``runtime_adapter.hook_answer``.
    pending_questions: list[str] = field(default_factory=list)
    # Monotonic instant after which the sandbox hook stops waiting. Compared before
    # sending rather than after: the hook has already given the worker its
    # unattended answer, so sending is a sandbox round-trip whose result is
    # discarded.
    pending_deadline: float | None = None
    turns: int = 0


@worker_manager_registry.register('pty_claude_code')
class PtyClaudeCodeSessionManager(SessionManager):
    """Manage interactive PTY Claude Code sessions for HITL recipes.

    Registered rather than merely constructible so that registration enforces the
    shared-resource convention: this manager takes the runtime's LLM handler
    instead of registering one of its own.
    """

    backend = 'pty_claude_code'
    product = 'Claude Code'
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
        self._sessions_by_name: dict[str, PtyClaudeCodeSession] = {}
        self._runtime: Runtime | None = None
        # Injected by the caller, or taken from the runtime in setup(). Never
        # built here: a manager that owns its handler claims the runtime's
        # broadcast llm_request channel, and a second such manager on the same
        # runtime makes every request run twice.
        self._handler: LabelledLLMRequestHandler | None = llm_handler
        self._owns_handler = False

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

    async def setup(self) -> None:
        if self._runtime is not None:
            return

        ctx = require_context()
        runtime = await ctx.get_runtime(self._runtime_name)
        for attempt in range(3):
            try:
                await runtime.ws_connect()
                break
            except Exception as exc:  # noqa: BLE001
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
            # Only the manager that acquired it may release it; an injected
            # handler belongs to whoever passed it in.
            self._owns_handler = True

        # One file per sandbox home, read by every session that starts after it.
        await place_user_memory(runtime, self._default_config, memory_path=self.memory_path)
        await place_worker_skills(runtime, self._default_config, skills_path=self.skills_path)

    async def start(
        self,
        *,
        name: str,
        workspace: str = '',
        metadata: Mapping[str, object] | None = None,
        config: Mapping[str, object] | None = None,
        **kwargs: object,
    ) -> PtyClaudeCodeSession:
        """Open a session and its labelled sandbox proxy.

        The proxy opens here rather than on the first turn, so a bad label or a
        sandbox problem surfaces at session start. Claude itself is still spawned
        lazily by the first turn.
        """
        if self._runtime is None:
            raise RuntimeError('PtyClaudeCodeSessionManager is not set up. Call setup() first.')
        labels.validate_name(name, kind='session name')
        if name in self._sessions_by_name:
            raise ValueError(f'PTY Claude Code session already exists: {name}')

        # The id is bound now and never rebound: the runtime-side registry keys
        # its live PTY on it, which is what makes turns reuse one Claude process.
        runtime_session_id = normalize_stream_json_session_id(
            str(uuid.uuid4()),
            namespace='pty-claude-code-session',
        )
        session = PtyClaudeCodeSession(
            name=name,
            session_id=runtime_session_id,
            backend=self.backend,
            metadata=dict(metadata or {}),
            workspace=workspace or self._default_working_dir,
            config={**self._default_config, **dict(config or {})},
        )

        await runtime_adapter.open_session(
            self._runtime,
            session_id=session.session_id,
            label=session.label,
            working_dir=session.workspace,
            config=session.config,
            extra_env=envs.session_env(session.config, session=session, harness_env=self.harness_env(session.config)),
        )

        self._sessions_by_name[name] = session
        return session

    def get(self, name: str) -> PtyClaudeCodeSession:
        return self._sessions_by_name[name]

    def session_of(self, session_id: str) -> PtyClaudeCodeSession | None:
        """The session a ``session_id`` belongs to, or None when it is not held."""
        return next(
            (session for session in self._sessions_by_name.values() if session.session_id == session_id),
            None,
        )

    async def send(self, session: Session, message: SessionMessage) -> TurnResult:
        if self._runtime is None or self._handler is None:
            raise RuntimeError('PtyClaudeCodeSessionManager is not set up. Call setup() first.')

        pty_session = _require_pty_session(session)

        # Refuse locally: these would either reset the session we are holding or
        # hang turn detection. Rejecting before the round trip keeps the sandbox
        # untouched and the turn counter unmoved.
        verdict = check_outgoing(
            message.content,
            allowed=self.commands,
            prefix=self.command_prefix,
            denied=self.denied_commands,
            unsupported=self.unsupported_commands,
        )
        if not verdict.allowed:
            logger.warning('[pty %s] blocked outgoing message: %s', pty_session.name, verdict.reason)
            return TurnResult(
                kind='refused',
                output='',
                session_id=pty_session.session_id,
                raw={'status': 'blocked_by_guard', 'reason': verdict.reason},
                metadata={
                    'message_kind': message.kind,
                    'name': pty_session.name,
                    'blocked': True,
                    'reason': verdict.reason,
                },
            )

        turn = pty_session.turns + 1

        answered_interaction_id = pty_session.pending_interaction_id
        if answered_interaction_id and self._answer_is_too_late(pty_session):
            return self._answer_missed(pty_session, message, turn=turn)

        # No trace scope: the session's proxy stamps its label onto every request,
        # so attribution does not depend on manager state and concurrent turns
        # need nothing from each other.
        started = time.monotonic()
        try:
            if answered_interaction_id:
                # Clear first: on failure the id is stale either way, and leaving it
                # set would route the next send into a dead interaction.
                questions = list(pty_session.pending_questions)
                pty_session.pending_interaction_id = None
                pty_session.pending_questions = []
                raw = await runtime_adapter.respond(
                    self._runtime,
                    session_id=pty_session.session_id,
                    interaction_id=answered_interaction_id,
                    answer=runtime_adapter.hook_answer(message.content, questions),
                )
            else:
                raw = await runtime_adapter.run_turn(
                    self._runtime,
                    session_id=pty_session.session_id,
                    prompt=message.content,
                    working_dir=pty_session.workspace,
                    config=pty_session.config,
                    extra_env=envs.session_env(
                        pty_session.config,
                        session=pty_session,
                        harness_env=self.harness_env(pty_session.config),
                    ),
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported as a terminal turn, not raised
            pty_session.turns = turn
            logger.warning('[pty %s] turn %d could not be run: %s', pty_session.name, turn, exc)
            return TurnResult(
                kind='terminal',
                output='',
                session_id=pty_session.session_id,
                raw={
                    'status': STATUS_UNREACHABLE,
                    'completed': False,
                    'reason': f'{type(exc).__name__}: {exc}',
                },
                metadata={
                    'message_kind': message.kind,
                    'name': pty_session.name,
                    'turn': turn,
                    'status': STATUS_UNREACHABLE,
                    'error': f'{type(exc).__name__}: {exc}',
                },
            )

        pty_session.turns = turn
        elapsed = time.monotonic() - started
        if answered_interaction_id and not runtime_adapter.answer_landed(raw):
            # The hook had already given up, so the worker acted on its own answer
            # and ours was discarded. Caught here as well as before sending because
            # the deadline we track is our arithmetic and this is upstream's fact.
            pty_session.turns = turn - 1
            return self._answer_missed(pty_session, message, turn=turn)
        return await self._build_result(
            pty_session,
            message,
            raw,
            turn=turn,
            answered_interaction_id=answered_interaction_id,
            seconds=elapsed,
        )

    @staticmethod
    def _answer_is_too_late(session: PtyClaudeCodeSession) -> bool:
        """Whether the hook waiting for this answer has already stopped waiting."""
        deadline = session.pending_deadline
        return deadline is not None and time.monotonic() >= deadline

    def _answer_missed(
        self,
        session: PtyClaudeCodeSession,
        message: SessionMessage,
        *,
        turn: int,
    ) -> TurnResult:
        """Report an answer the worker never received.

        ``refused`` rather than ``completed``: the session is untouched and the next
        send works, which is what separates this from a lost session. Reported as a
        turn that did not happen -- the worker ran no turn on our behalf -- so the
        pending state is cleared here and the next message starts a fresh one.
        """
        logger.info(
            '[pty %s] answer to interaction %s arrived after the hook stopped waiting',
            session.name,
            session.pending_interaction_id or '(cleared)',
        )
        session.pending_interaction_id = None
        session.pending_questions = []
        session.pending_deadline = None
        return TurnResult(
            kind='refused',
            output='',
            session_id=session.session_id,
            raw={'status': STATUS_ANSWER_MISSED, 'completed': False, 'reason': ANSWER_MISSED_REASON},
            metadata={
                'message_kind': message.kind,
                'name': session.name,
                'turn': turn,
                'status': STATUS_ANSWER_MISSED,
                'reason': ANSWER_MISSED_REASON,
            },
        )

    async def _build_result(
        self,
        session: PtyClaudeCodeSession,
        message: SessionMessage,
        raw: Mapping[str, object],
        *,
        turn: int,
        answered_interaction_id: str | None,
        seconds: float,
    ) -> TurnResult:
        output = str(raw.get('output') or '')
        status = str(raw.get('status') or '')

        user_input_request: UserInputRequest | None = await runtime_adapter.build_hook_request(
            raw, session_id=session.session_id, turn=turn
        )
        if user_input_request is not None:
            session.pending_interaction_id = runtime_adapter.pending_interaction_id(raw)
            session.pending_questions = runtime_adapter.pending_questions(raw)
            wait_seconds = runtime_adapter.pending_wait_seconds(raw)
            session.pending_deadline = None if wait_seconds is None else time.monotonic() + wait_seconds

        session_lost = runtime_adapter.session_died(raw)
        if session_lost:
            # The runtime already closed and de-registered the PTY, so the next
            # send would silently start a fresh Claude with no history. Log it
            # here because the result alone only says the kind.
            logger.warning(
                '[pty %s] turn %d ended as %r; runtime-side session was torn down',
                session.name,
                turn,
                status or 'unknown',
            )

        return TurnResult(
            kind=_turn_kind(session_lost=session_lost, awaiting=user_input_request is not None),
            output=output,
            session_id=str(raw.get('session_id') or session.session_id),
            raw=dict(raw),
            user_input_request=user_input_request,
            cost=account_worker_turn(session, seconds=seconds),
            metadata={
                'message_kind': message.kind,
                'name': session.name,
                'turn': turn,
                'status': status,
                'turn_id': raw.get('turn_id'),
                'answered_interaction_id': answered_interaction_id or '',
            },
        )

    def report(self, result: TurnResult, *, detail: Detail = DEFAULT_DETAIL) -> str:
        """Project the turn from the event stream this backend records.

        A human at the interactive TUI watches text and tool calls scroll past in one
        order, so the steps keep that order: the events arrive interleaved and each
        tool call is named where it happened.
        """
        _, events, _ = runtime_adapter.perception(dict(result.raw))
        return append_cost(
            report_from_steps(
                steps_with_detail(events),
                output=result.output,
                question=result.question,
                empty_note='This turn recorded no message events; only its combined output is available.',
                detail=detail,
            ),
            result.cost,
        )

    def traces(self, session: Session) -> dict[str, object]:
        """This session's own trajectories, from the run-level tracker.

        Requires a run-level ``chat_tracker`` in ``branch_last`` mode;
        ``global_first`` exports bare message lists with nowhere to carry a label.
        """
        return labels.filter_by_label(labels.run_level_trajectories(), session.label)

    def unattributed_requests(self) -> int:
        """Requests that arrived with no resolvable label. Should be zero."""
        if self._handler is None:
            return 0
        return self._handler.unattributed_requests

    async def close(self, session: Session) -> None:
        if self._runtime is not None:
            try:
                await runtime_adapter.close(self._runtime, session_id=session.session_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning('Failed to close PTY session %s: %s', session.session_id, exc)
        self._sessions_by_name.pop(session.name, None)

    async def close_session(self, name: str) -> None:
        await self.close(self.get(name))

    async def close_all(self) -> None:
        for session in list(self._sessions_by_name.values()):
            await self.close(session)

    async def shutdown(self) -> None:
        if self._runtime is not None:
            try:
                await runtime_adapter.close_all(self._runtime)
            except Exception as exc:  # noqa: BLE001
                logger.warning('Failed to close PTY runtime sessions: %s', exc)
        # Release only what this manager acquired. An injected handler may still
        # be serving another manager on the same runtime.
        if self._owns_handler and self._runtime is not None:
            release_llm_handler(self._runtime)
            self._owns_handler = False
        self._handler = None
        self._runtime = None
        self._sessions_by_name.clear()


def _require_pty_session(session: Session) -> PtyClaudeCodeSession:
    if not isinstance(session, PtyClaudeCodeSession):
        raise TypeError(f'Expected PtyClaudeCodeSession, got {type(session).__name__}')
    return session


def _turn_kind(*, session_lost: bool, awaiting: bool) -> TurnKind:
    """How the turn ended.

    A torn-down session outranks a pending question: the question can no longer be
    answered, so reporting it as answerable would send the next message into a dead
    interaction.
    """
    if session_lost:
        return 'terminal'
    return 'needs_input' if awaiting else 'completed'


__all__ = [
    'PtyClaudeCodeSession',
    'PtyClaudeCodeSessionManager',
]
