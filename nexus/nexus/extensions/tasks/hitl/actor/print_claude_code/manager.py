"""Claude Code print-mode session manager.

One ``claude --print`` process per turn, resumed by session id. Each session owns
its own sandbox proxy, which fixes its trace label and so lets sessions run
concurrently -- ``SessionManager.run_turns`` drives that.
"""

import asyncio
import logging
import time
import uuid

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from nexus.core import UserInputRequest, require_context
from nexus.extensions.agents.installed_common.multiturn import normalize_stream_json_session_id
from nexus.extensions.tasks.hitl.actor import envs
from nexus.extensions.tasks.hitl.actor.base import (
    STATUS_UNREACHABLE,
    Session,
    SessionManager,
    SessionMessage,
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
from nexus.extensions.tasks.hitl.actor.print_claude_code.backend import (
    DEFAULT_CC_TURN_TIMEOUT_SECONDS,
    LabelledPrintBackend,
)
from nexus.extensions.tasks.hitl.actor.print_claude_code.rendering import steps_with_detail
from nexus.extensions.tasks.hitl.actor.print_claude_code.spec import (
    BUILTIN_COMMANDS,
    COMMAND_PREFIX,
    DENIED_COMMANDS,
    MEMORY_PATH,
    SKILLS_PATH,
    UNSUPPORTED_COMMANDS,
    worker_env,
)
from nexus.extensions.tasks.hitl.actor.registry import worker_manager_registry
from nexus.extensions.tasks.hitl.actor.rendering import (
    DEFAULT_DETAIL,
    Detail,
    render_questions,
    report_from_steps,
)
from nexus.extensions.tasks.hitl.actor.resources import acquire_llm_handler, release_llm_handler
from nexus.extensions.tasks.hitl.trace import labels
from nexus.extensions.tasks.hitl.utils import coerce
from nexus.runtime import Runtime


logger = logging.getLogger(__name__)


@dataclass
class PrintClaudeCodeSession(Session):
    """One resumable print-mode session."""

    workspace: str = '/workspace'
    config: dict[str, object] = field(default_factory=dict)
    pending_parent_tool_use_id: str | None = None
    pending_questions: list[str] = field(default_factory=list)
    """The questions the worker is blocked on.

    Claude Code keys its answers by question text, so answering needs the text the
    deferring turn carried.
    """
    turns: int = 0


@worker_manager_registry.register('print_claude_code')
class PrintClaudeCodeSessionManager(SessionManager):
    """Drive many ``claude --print`` sessions, concurrently if asked."""

    backend = 'print_claude_code'
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
        self._sessions_by_name: dict[str, PrintClaudeCodeSession] = {}
        self._runtime: Runtime | None = None
        self._handler = llm_handler
        self._owns_handler = False
        self._backend = LabelledPrintBackend(
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
    ) -> PrintClaudeCodeSession:
        """Open a session and its sandbox proxy.

        The proxy opens here rather than on the first turn, so a bad label or a
        sandbox problem surfaces at session start.
        """
        if self._runtime is None:
            raise RuntimeError('PrintClaudeCodeSessionManager is not set up. Call setup() first.')
        labels.validate_name(name, kind='session name')
        if name in self._sessions_by_name:
            raise ValueError(f'print Claude Code session already exists: {name}')

        session = PrintClaudeCodeSession(
            name=name,
            session_id=normalize_stream_json_session_id(str(uuid.uuid4()), namespace='claude-code-session'),
            backend=self.backend,
            metadata=dict(metadata or {}),
            workspace=workspace or self._default_working_dir,
            config={**self._default_config, **dict(config or {})},
        )

        await self._backend.open_session(
            self._runtime,
            session_id=session.session_id,
            label=session.label,
            working_dir=session.workspace,
        )

        self._sessions_by_name[name] = session
        return session

    def get(self, name: str) -> PrintClaudeCodeSession:
        return self._sessions_by_name[name]

    def session_of(self, session_id: str) -> PrintClaudeCodeSession | None:
        """The session a ``session_id`` belongs to, or None when it is not held."""
        return next(
            (session for session in self._sessions_by_name.values() if session.session_id == session_id),
            None,
        )

    # -- turns -------------------------------------------------------------

    async def send(self, session: Session, message: SessionMessage) -> TurnResult:
        """Run one turn. Safe to call concurrently for different sessions."""
        if self._runtime is None or self._handler is None:
            raise RuntimeError('PrintClaudeCodeSessionManager is not set up. Call setup() first.')

        print_session = _require_print_session(session)

        verdict = check_outgoing(
            message.content,
            allowed=self.commands,
            prefix=self.command_prefix,
            denied=self.denied_commands,
            unsupported=self.unsupported_commands,
        )
        if not verdict.allowed:
            logger.warning('[print %s] blocked outgoing message: %s', print_session.name, verdict.reason)
            return TurnResult(
                kind='refused',
                session_id=print_session.session_id,
                raw={'status': 'blocked_by_guard', 'reason': verdict.reason},
                metadata={
                    'message_kind': message.kind,
                    'name': print_session.name,
                    'blocked': True,
                    'reason': verdict.reason,
                },
            )

        turn = print_session.turns + 1

        pending_tool_use_id = print_session.pending_parent_tool_use_id
        pending_answers: dict[str, str] | None = None
        if pending_tool_use_id:
            answers = message.metadata.get('answers')
            pending_answers = (
                {str(key): str(value) for key, value in answers.items()}
                if isinstance(answers, dict)
                else _answers_for(print_session.pending_questions, message.content)
            )
            # Cleared before the turn; on failure the id is stale either way.
            print_session.pending_parent_tool_use_id = None
            print_session.pending_questions = []

        started = time.monotonic()
        try:
            result = await self._backend.send_turn(
                self._runtime,
                session_id=print_session.session_id,
                prompt=message.content,
                working_dir=print_session.workspace,
                timeout=_turn_timeout(print_session.config),
                extra_env=envs.session_env(
                    print_session.config,
                    session=print_session,
                    harness_env=self.harness_env(print_session.config),
                ),
                extra_cli_args=coerce.optional_str_list(print_session.config.get('extra_cli_args')),
                pending_tool_use_id=pending_tool_use_id,
                pending_answers=pending_answers,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported as a lost session, not raised
            print_session.turns = turn
            logger.warning('[print %s] turn %d could not be run: %s', print_session.name, turn, exc)
            return TurnResult(
                kind='terminal',
                output='',
                session_id=print_session.session_id,
                raw={'status': STATUS_UNREACHABLE, 'reason': f'{type(exc).__name__}: {exc}'},
                metadata={
                    'message_kind': message.kind,
                    'name': print_session.name,
                    'turn': turn,
                    'status': STATUS_UNREACHABLE,
                    'error': f'{type(exc).__name__}: {exc}',
                },
            )

        print_session.turns = turn
        elapsed = time.monotonic() - started
        raw = {str(key): value for key, value in result.items()}
        output = str(result.get('output') or '')

        user_input_request: UserInputRequest | None = None
        deferred = result.get('deferred_tool')
        if isinstance(deferred, dict):
            print_session.pending_parent_tool_use_id = str(deferred.get('tool_use_id', ''))
            print_session.pending_questions = _deferred_questions(deferred)
            user_input_request = _user_input_request(
                deferred=deferred,
                session_id=print_session.session_id,
                turn=turn,
            )

        return TurnResult(
            kind='needs_input' if user_input_request is not None else 'completed',
            output=output,
            session_id=str(result.get('session_id') or print_session.session_id),
            raw=raw,
            user_input_request=user_input_request,
            cost=account_worker_turn(print_session, seconds=elapsed),
            metadata={
                'message_kind': message.kind,
                'name': print_session.name,
                'turn': turn,
                'label': print_session.label,
            },
        )

    # -- reporting ---------------------------------------------------------

    def report(self, result: TurnResult, *, detail: Detail = DEFAULT_DETAIL) -> str:
        """Project the turn from the steps this backend records.

        Claude Code's stream-json output carries every message it sent and every tool
        it called, interleaved, so the turn is reported in the order it happened and
        the last message is its claim.
        """
        return append_cost(
            report_from_steps(
                steps_with_detail(result.raw.get('steps')),
                output=result.output,
                question=result.question,
                empty_note='This turn produced no separable messages; only its combined output is available.',
                detail=detail,
            ),
            result.cost,
        )

    # -- traces ------------------------------------------------------------

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

    # -- teardown ----------------------------------------------------------

    async def close(self, session: Session) -> None:
        print_session = _require_print_session(session)
        if self._runtime is not None:
            await self._backend.close_session(self._runtime, session_id=print_session.session_id)
        self._sessions_by_name.pop(print_session.name, None)

    async def close_session(self, name: str) -> None:
        await self.close(self.get(name))

    async def close_all(self) -> None:
        for session in list(self._sessions_by_name.values()):
            await self.close(session)

    async def shutdown(self) -> None:
        if self._runtime is not None:
            await self._backend.shutdown(self._runtime)
            if self._owns_handler:
                release_llm_handler(self._runtime)
                self._owns_handler = False
        self._handler = None
        self._runtime = None
        self._sessions_by_name.clear()


def _require_print_session(session: Session) -> PrintClaudeCodeSession:
    if not isinstance(session, PrintClaudeCodeSession):
        raise TypeError(f'Expected PrintClaudeCodeSession, got {type(session).__name__}')
    return session


def _turn_timeout(config: Mapping[str, object]) -> float:
    """Per-turn timeout, falling back to the default for anything unusable."""
    return coerce.optional_float(config.get('timeout'), default=float(DEFAULT_CC_TURN_TIMEOUT_SECONDS))


def _deferred_questions(deferred: Mapping[str, object]) -> list[str]:
    """The question texts an ``AskUserQuestion`` deferred on."""
    tool_input = deferred.get('input')
    questions = cast(Mapping[str, Any], tool_input).get('questions') if isinstance(tool_input, dict) else None
    if not isinstance(questions, list):
        return []
    return [
        str(question.get('question', f'question_{index}'))
        for index, question in enumerate(questions)
        if isinstance(question, dict)
    ]


def _answers_for(questions: Sequence[str], content: str) -> dict[str, str]:
    """The answers dict Claude Code expects, keyed by question text.

    With several questions only the first gets the reply; the rest are marked as
    covered by it, because the orchestrator answers a turn rather than a question.
    """
    if not questions:
        return {content: content}
    return {text: content if index == 0 else '(answered above)' for index, text in enumerate(questions)}


def _user_input_request(*, deferred: Mapping[str, object], session_id: str, turn: int) -> UserInputRequest:
    """Render a deferred AskUserQuestion as a request for the orchestrator.

    ``hook`` carries the tool input whole, which is what ``TurnResult.question``
    renders the options from.
    """
    tool_input = deferred.get('input')
    questions = cast(Mapping[str, Any], tool_input).get('questions') if isinstance(tool_input, Mapping) else None
    return UserInputRequest(
        source='hook',
        hint=render_questions(questions, agent='Claude Code') or 'Claude Code asks for user input.',
        session_id=session_id,
        message={},
        events=[],
        hook=dict(deferred),
        metadata={'turn': turn, 'agent': 'Claude Code'},
    )


__all__ = ['PrintClaudeCodeSession', 'PrintClaudeCodeSessionManager']
