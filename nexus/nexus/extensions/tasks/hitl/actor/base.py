"""Actor session interfaces for HITL collaboration.

A worker backend implements :class:`SessionManager`. Sessions are attributed by the
label their sandbox proxy stamps on each LLM request, so one run-level chat tracker
holds every conversation and concurrent sessions need no per-session state.
"""

import asyncio
import logging

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from nexus.core import UserInputRequest
from nexus.extensions.tasks.hitl.actor.rendering import DEFAULT_DETAIL, Detail, render_questions
from nexus.extensions.tasks.hitl.trace.labels import worker_label


logger = logging.getLogger(__name__)


TurnKind = Literal['completed', 'needs_input', 'terminal', 'refused']
"""How one turn ended.

* ``completed`` -- the worker ran and finished its turn.
* ``needs_input`` -- the worker is blocked on a question and the next message is
  routed as the answer to it.
* ``terminal`` -- the session cannot serve another turn. Sending again would
  either fail the same way or silently reach a fresh agent with no history.
* ``refused`` -- the message was not delivered and the session is untouched, so
  the next send works. A rewritten message is worth trying.

``refused`` and ``terminal`` are the two that a driver must not confuse: retrying
is right for one and pointless for the other.
"""

# Status for a turn the host could not run at all, because the call into the sandbox
# never returned a result.
STATUS_UNREACHABLE = 'runtime_unreachable'

# Status for an answer that reached a worker no longer waiting for one: it answered
# its own question and carried on, so ours was discarded.
STATUS_ANSWER_MISSED = 'answer_missed'


@dataclass
class Session:
    """One worker conversation, and the label its requests carry."""

    name: str
    session_id: str
    backend: str
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """The trace label every request from this session carries."""
        return worker_label(self.name)


@dataclass(frozen=True)
class SessionMessage:
    content: str
    kind: str = 'message'
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnCost:
    """One worker turn's wall-clock, and how much context the worker is carrying."""

    duration_s: float
    context_tokens: int


@dataclass(frozen=True)
class TurnResult:
    """What one turn produced, in backend-neutral terms.

    ``raw`` carries the backend's own turn material unchanged; a backend's projector
    reads it to build the report (see :mod:`..actor.rendering`).
    """

    kind: TurnKind
    output: str = ''
    session_id: str = ''
    raw: dict[str, object] = field(default_factory=dict)
    user_input_request: UserInputRequest | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    cost: TurnCost | None = None

    @property
    def awaiting_answer(self) -> bool:
        """The worker asked a question and is blocked until it is answered."""
        return self.kind == 'needs_input'

    @property
    def session_lost(self) -> bool:
        """The session cannot serve another turn."""
        return self.kind == 'terminal'

    @property
    def delivered(self) -> bool:
        """The message reached the worker, whatever came of it afterwards."""
        return self.kind != 'refused'

    @property
    def has_output(self) -> bool:
        return bool(self.output.strip())

    @property
    def question(self) -> str:
        """The worker's pending question, or empty when it is not blocked.

        Rendered from the deferred tool's own input when it carries questions, so
        the options and their trade-offs travel with the question. The ``hint``
        both backends build lists the question texts only.
        """
        if self.user_input_request is None:
            return ''
        hook = self.user_input_request.hook
        if isinstance(hook, Mapping):
            tool_input = hook.get('input')
            if isinstance(tool_input, Mapping):
                rendered = render_questions(tool_input.get('questions'))
                if rendered:
                    return rendered
        return self.user_input_request.hint or 'Worker requested input.'


@dataclass(frozen=True)
class BatchTurnResult:
    """One session's finished turn out of a batch, plus what the batch is still doing.

    ``inflight`` names the sessions still running when this turn finished.
    """

    session: Session
    result: TurnResult
    inflight: frozenset[str] = frozenset()

    @property
    def kind(self) -> TurnKind:
        return self.result.kind


class SessionManager(ABC):
    """Backend wrapper for multi-turn actor sessions.

    A manager takes the runtime's shared ``llm_handler``, which
    ``actor.resources.acquire_llm_handler`` hands out per runtime.
    """

    backend: str = ''
    """The name this manager is reached by. Every concrete manager sets it."""

    product: str = ''
    """What a person would call this tool, e.g. ``'Claude Code'``.

    Reaches the user simulator's prompt. ``backend`` names an implementation and mixes
    the product with how it is driven (``pty_claude_code`` against
    ``print_claude_code``).
    """

    memory_path: str = ''
    """Where this harness reads its user memory, relative to the sandbox home.

    Claude Code reads ``.claude/CLAUDE.md``; another harness reading ``AGENTS.md``
    declares that instead. Empty means the harness has no such file, and
    ``worker_config['user_memory']`` is not written.
    """

    skills_path: str = ''
    """Where this harness reads installed skills, relative to the sandbox home.

    Claude Code reads ``.claude/skills``, one directory per skill named as the skill
    is. Empty means the harness has no such directory, and a run naming skill
    directories is refused rather than writing them where nothing reads them.
    """

    command_prefix: str = ''
    """What a message must start with to run one of the harness's own commands.

    Claude Code takes ``/name`` as the first token of an ordinary prompt, so its
    managers declare ``'/'``. Only the first token is the command; the rest of the
    message is its input. Empty means the harness has no such form, and the user
    simulator is told nothing about commands.
    """

    builtin_commands: Mapping[str, str] = MappingProxyType({})
    """The harness's own commands: name -> what it does, without the prefix.

    Only commands that produce a reply every time belong here; one that mutates the
    client instead leaves the turn waiting out its timeout (see ``actor.input_spec``).
    """

    denied_commands: frozenset[str] = frozenset()
    """Commands that discard, detach, or relocate the session this manager holds.

    Refused by ``input_spec.check_outgoing``. Empty for a harness with no such command.
    """

    unsupported_commands: frozenset[str] = frozenset()
    """Commands that produce no assistant turn, so nothing signals completion.

    Refused by ``input_spec.check_outgoing``. Empty for a harness with no such command.
    """

    def __init__(self, **kwargs: object) -> None:
        """The construction args ``factory.make_worker_manager`` forwards.

        A backend names the ones it reads: ``default_working_dir``,
        ``default_config``, ``openai_client_name`` and its own ``llm_handler``.
        """
        del kwargs
        super().__init__()

    def harness_env(self, config: Mapping[str, object]) -> dict[str, str]:
        """The environment variables this harness's own process needs.

        ``config`` is the session's ``worker_config``. Called by
        ``envs.session_env``, which adds the host credentials around it. Returns
        nothing by default.
        """
        del config
        return {}

    @property
    def commands(self) -> Mapping[str, str]:
        """Every command this manager accepts: its harness's own plus this run's."""
        return {**self.builtin_commands, **self.run_commands}

    @property
    def run_commands(self) -> Mapping[str, str]:
        """The commands this run installed. Empty for a manager that installs none."""
        return {}

    @abstractmethod
    async def setup(self) -> None:
        """Prepare shared backend resources before sessions are used."""

    @abstractmethod
    async def start(
        self,
        *,
        name: str,
        metadata: Mapping[str, object] | None = None,
        **kwargs: object,
    ) -> Session:
        """Start an actor session."""

    @abstractmethod
    async def send(self, session: Session, message: SessionMessage) -> TurnResult:
        """Send one message to an actor session."""

    async def run_turns(self, turns: Sequence[tuple[Session, SessionMessage]]) -> AsyncIterator[BatchTurnResult]:
        """Send to several sessions at once, yielding each turn as it finishes.

        Results come in completion order, so a caller can act on the first one while
        the rest go on: read its report, then ``send`` to that session again. A send
        that raises is reported as a ``terminal`` turn. Raises ValueError for a session
        named twice.
        """
        names = [session.name for session, _ in turns]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f'a session can take one turn at a time; addressed twice: {repeated}')

        pending = {asyncio.ensure_future(self.send(session, message)): session for session, message in turns}
        remaining = set(names)
        while pending:
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                session = pending.pop(task)
                remaining.discard(session.name)
                yield BatchTurnResult(
                    session=session,
                    result=_turn_result_of(task, name=session.name),
                    inflight=frozenset(remaining),
                )

    @abstractmethod
    def session_of(self, session_id: str) -> Session | None:
        """The session a ``session_id`` belongs to, or None when it is not held."""

    @abstractmethod
    def report(self, result: TurnResult, *, detail: Detail = DEFAULT_DETAIL) -> str:
        """Render one turn as the tagged sections its reader is given.

        Each backend reads its own material out of ``result.raw`` and fills the
        sections it can observe: a print-mode turn splits on the assistant messages
        it recorded, an interactive turn on its event stream, which also names the
        tools used. The section vocabulary is in :mod:`..actor.rendering`.
        """

    @abstractmethod
    def traces(self, session: Session) -> dict[str, object]:
        """Return this session's own trajectories from the run-level export."""

    @abstractmethod
    async def close(self, session: Session) -> None:
        """Close one session."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Release shared backend resources."""


def _turn_result_of(task: 'asyncio.Future[TurnResult]', *, name: str) -> TurnResult:
    """The task's result, or a terminal turn describing why there is none."""
    error = task.exception()
    if error is None:
        return task.result()
    logger.warning('session %s could not take its turn: %s', name, error)
    return TurnResult(
        kind='terminal',
        session_id='',
        raw={'status': STATUS_UNREACHABLE, 'reason': f'{type(error).__name__}: {error}'},
        metadata={'name': name, 'status': STATUS_UNREACHABLE, 'error': f'{type(error).__name__}: {error}'},
    )


__all__ = [
    'STATUS_ANSWER_MISSED',
    'STATUS_UNREACHABLE',
    'BatchTurnResult',
    'Session',
    'SessionManager',
    'SessionMessage',
    'TurnKind',
    'TurnCost',
    'TurnResult',
]
