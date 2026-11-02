"""A simulated user driving one worker harness, as a nexus agent function.

Each turn the user calls a tool: it sends the worker a message, looks at the work
itself, or accepts. Accepting releases the next requirement, and accepting the last
one ends the run. :class:`RunState` holds everything the run accumulates and becomes
the output.

``requirements`` is what the user asks for, one entry at a time; a task with a single
``user_prompt`` gets a queue of one. ``worker_backend`` names the harness, reached
only through ``SessionManager`` and ``TurnResult``. ``hooks`` is where a topic puts its
own logic -- see :mod:`..user_agent.hooks`.
"""

import logging
import time

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import Field

from nexus.core import ToolResponse, agent_registry, require_context
from nexus.core.types.message import ChatCompletionMessageToolCall
from nexus.extensions.agents.installed_common import InstalledAgentOutput, InstalledNodeCliInput
from nexus.extensions.tasks.hitl.actor.base import (
    STATUS_ANSWER_MISSED,
    Session,
    SessionManager,
    SessionMessage,
    TurnResult,
)
from nexus.extensions.tasks.hitl.actor.factory import make_worker_manager
from nexus.extensions.tasks.hitl.actor.memory import memory_text
from nexus.extensions.tasks.hitl.actor.rendering import WORKER_REPORT_PROTOCOL
from nexus.extensions.tasks.hitl.trace import labels
from nexus.extensions.tasks.hitl.trace.usage import HITLUsageProcessor
from nexus.extensions.tasks.hitl.user_agent import queue as queue_mod
from nexus.extensions.tasks.hitl.user_agent import skills as skills_mod
from nexus.extensions.tasks.hitl.user_agent.compact import (
    DEFAULT_COMPACT_PCT,
    SUMMARY_PROMPT,
    Budget,
    compact,
)
from nexus.extensions.tasks.hitl.user_agent.dispatch import (
    Dispatcher,
    ExtendedToolResponse,
    Handler,
    call_id,
    call_name,
    check_handlers,
    content_of,
    read_response,
    tool_message,
)
from nexus.extensions.tasks.hitl.user_agent.hooks import (
    HookEvent,
    RequiredHookError,
    RunHook,
    resolve_hooks,
)
from nexus.extensions.tasks.hitl.user_agent.language import (
    DEFAULT_LANGUAGE,
    NaturalLanguage,
    detect,
)
from nexus.extensions.tasks.hitl.user_agent.personas import get_persona, load_persona_dirs
from nexus.extensions.tasks.hitl.user_agent.prompts import (
    NUDGE,
    append_reminder,
    build_opening_message,
    build_requirement_message,
    build_system_prompt,
    environment_facts,
    handover_reminder,
    language_instruction,
    reminder,
    reminder_message,
    user_memory_section,
    worker_command_lines,
    worker_skills_section,
)
from nexus.extensions.tasks.hitl.user_agent.queue import Requirements
from nexus.extensions.tasks.hitl.user_agent.rating import ExperienceRating, rate_experience
from nexus.extensions.tasks.hitl.user_agent.reminders import (
    HEAVY_EVERY,
    LIGHT_EVERY,
    LIGHT_TEXT,
    Reminders,
)
from nexus.extensions.tasks.hitl.user_agent.skills import catalogue
from nexus.extensions.tasks.hitl.user_agent.tools import toolset
from nexus.extensions.tasks.hitl.user_agent.tools.orchestration import (
    EMPTY_MESSAGE_REFUSAL,
    INCOMPLETE_ACCEPT_REFUSAL,
    LANGUAGE_REFUSAL,
    LAST_ACCEPT_REMINDER,
    MISSING_DRAFT_REFUSAL,
    NOT_DELIVERED,
    PENDING_QUESTION_REFUSAL,
    SENT_TEMPLATE,
    SESSION_LOST_NOTICE,
    SESSION_LOST_REASON,
)
from nexus.trace.context import get_processor


logger = logging.getLogger(__name__)


AGENT_NAME = 'hitl/agent_with_user'
DEFAULT_BACKEND = 'print_claude_code'
WORKER_NAME = 'worker'

# The label every host-side user turn is recorded under. Worker sessions label
# themselves in the sandbox.
USER_LABEL = labels.user_label()

# Tools the user looks with, answered by one handler.
LOOKING_TOOLS = frozenset({'read', 'bash', 'str_replace_editor'})


@dataclass
class RunState:
    """Everything one run accumulates, and the live objects it acts through.

    The agent builds it before the worker starts, puts it on every hook event, and
    reads the output's fields off it.
    """

    queue: Requirements

    manager: SessionManager | None = None
    session: Session | None = None

    natural_language: NaturalLanguage = DEFAULT_LANGUAGE
    """The language the user writes to the worker in, read from the task.

    Not the programming language of the work.
    """

    stop_reason: str = ''
    worker_rounds: int = 0
    messages_sent: int = 0
    awaiting_answer: bool = False
    accepted: list[dict[str, str]] = field(default_factory=list)

    handovers: int = 0
    """Times work has come back for the requirement now current. Reset on accept."""

    reminded_of_last: bool = False
    """Whether the reminder before the final acceptance has been given. Cleared by a
    message to the worker."""

    turns: int = 0
    idle_turns: int = 0
    compactions: int = 0
    failures: int = 0
    stopped: bool = False
    error: str = ''
    experience: ExperienceRating = field(default_factory=ExperienceRating)

    hook_results: dict[str, Any] = field(default_factory=dict)
    hook_failures: list[str] = field(default_factory=list)

    @property
    def session_lost(self) -> bool:
        return self.stop_reason == SESSION_LOST_REASON

    @property
    def status(self) -> str:
        """How the run ended, most specific first."""
        if self.error:
            return 'error'
        if self.session_lost:
            return 'session_lost'
        return 'completed' if self.stopped else 'max_turns'

    @property
    def succeeded(self) -> bool:
        return not self.error and self.stopped and not self.session_lost


@dataclass(frozen=True, kw_only=True)
class RunStart(HookEvent):
    """The worker is up and the conversation has not started."""

    name: ClassVar[str] = f'{AGENT_NAME}:run.start'

    state: RunState


@dataclass(frozen=True, kw_only=True)
class WorkerDone(HookEvent):
    """A worker turn has come back, and the run's counters include it.

    ``result.kind`` says which way it went: ``completed`` with work handed back,
    ``needs_input`` waiting on an answer, ``refused`` never delivered, ``terminal``
    with the session gone. The properties ``delivered``, ``awaiting_answer``,
    ``has_output`` and ``session_lost`` read those.
    """

    name: ClassVar[str] = f'{AGENT_NAME}:worker.done'

    state: RunState
    worker: str
    result: TurnResult


@dataclass(frozen=True, kw_only=True)
class RunFinish(HookEvent):
    """The turn loop has ended and the worker is about to be torn down."""

    name: ClassVar[str] = f'{AGENT_NAME}:run.finish'

    state: RunState


EVENTS: tuple[type[HookEvent], ...] = (RunStart, WorkerDone, RunFinish)
"""What a hook may subscribe to on this agent."""


def accepted_text(accepted: Sequence[Mapping[str, str]]) -> str:
    """One line per requirement the user accepted, in the words it used."""
    return '\n'.join(entry['summary'] for entry in accepted if entry['summary'])


def build_handlers(state: RunState, *, hooks: Sequence[RunHook] = ()) -> dict[str, Handler]:
    """The handlers this agent needs: delivering, accepting, and looking."""

    async def send_message(
        call: ChatCompletionMessageToolCall, response: ToolResponse, args: dict[str, Any]
    ) -> ExtendedToolResponse:
        message = str(args.get('message') or '')
        if not message.strip():
            return _refused(call, 'send_message', EMPTY_MESSAGE_REFUSAL, response, tag='error')
        if not str(args.get('draft') or '').strip():
            return _refused(call, 'send_message', MISSING_DRAFT_REFUSAL, response)
        addressed = str(args.get('to') or '')
        if addressed and addressed != WORKER_NAME:
            detail = f'unknown worker {addressed!r}. There is: {WORKER_NAME}'
            return _refused(call, 'send_message', detail, response, tag='error')
        if not state.natural_language.accepts(message):
            detail = LANGUAGE_REFUSAL.format(name=state.natural_language.name)
            return _refused(call, 'send_message', detail, response)

        assert state.manager is not None
        assert state.session is not None
        result = await state.manager.send(state.session, SessionMessage(content=message, kind='message'))
        if result.delivered:
            state.worker_rounds += 1
            state.messages_sent += 1
            state.awaiting_answer = result.awaiting_answer
            state.reminded_of_last = False
        elif result.raw.get('status') == STATUS_ANSWER_MISSED:
            # The reason is already addressed to the user; resending does not help.
            state.awaiting_answer = False
        # Fired once the counters describe this round, whichever way it went.
        await notify_hooks(hooks, WorkerDone(state=state, worker=WORKER_NAME, result=result), state)

        if not result.delivered:
            reason = str(result.raw.get('reason') or 'the backend refused to send it')
            logger.info('message not delivered: %s', reason)
            detail = reason if result.raw.get('status') == STATUS_ANSWER_MISSED else f'{NOT_DELIVERED}\n\n{reason}'
            return _refused(call, 'send_message', detail, response, tag='error')

        claims_done = not result.awaiting_answer and result.has_output
        if claims_done:
            state.handovers += 1
        content = SENT_TEMPLATE.format(text=message, report=state.manager.report(result))
        if result.session_lost:
            # The runtime closed and de-registered the session. The run ends here.
            state.stopped = True
            state.stop_reason = SESSION_LOST_REASON
            logger.warning('worker session lost on round %d; ending the run', state.worker_rounds)
            return ExtendedToolResponse(
                tool_call_id=call_id(call),
                name='send_message',
                content=append_reminder(content, SESSION_LOST_NOTICE),
                response=response,
            )
        handover = handover_reminder(request=state.queue.position, handover=state.handovers)
        return ExtendedToolResponse(
            tool_call_id=call_id(call),
            name='send_message',
            content=content,
            response=response,
            messages=(
                reminder_message(language_instruction(state.natural_language.name)),
                reminder_message(handover) if claims_done else {},
            ),
        )

    async def accept(
        call: ChatCompletionMessageToolCall, response: ToolResponse, args: dict[str, Any]
    ) -> ExtendedToolResponse:
        if state.awaiting_answer:
            return _refused(call, 'accept', PENDING_QUESTION_REFUSAL, response)

        reason = str(args.get('reason') or 'satisfied')
        summary = str(args.get('summary') or '')
        if reason == 'satisfied' and not str(args.get('checked') or '').strip():
            return _refused(call, 'accept', INCOMPLETE_ACCEPT_REFUSAL, response)
        if reason == 'satisfied' and state.queue.on_last and not state.reminded_of_last:
            state.reminded_of_last = True
            return _refused(call, 'accept', LAST_ACCEPT_REMINDER, response, tag='held')

        # The queue goes first; it raises on an accept it has no requirement for.
        following = state.queue.accept(summary)
        state.accepted.append({'reason': reason, 'summary': summary})

        if following is None:
            state.stopped = True
            state.stop_reason = reason
            return ExtendedToolResponse(
                tool_call_id=call_id(call),
                name='accept',
                content='All requirements accepted.',
                response=response,
            )
        state.handovers = 0
        return ExtendedToolResponse(
            tool_call_id=call_id(call),
            name='accept',
            content='Current requirements accepted, and there are more requirements coming.',
            response=response,
            messages=({'role': 'user', 'content': build_requirement_message(following)},),
        )

    async def inspect(
        call: ChatCompletionMessageToolCall, response: ToolResponse, args: dict[str, Any]
    ) -> ExtendedToolResponse:
        """Answer a looking tool with what it saw, keeping any non-text content part."""
        del args
        return ExtendedToolResponse(
            tool_call_id=call_id(call), name=call_name(call), content=content_of(response), response=response
        )

    handlers: dict[str, Handler] = {'send_message': send_message, 'accept': accept}
    handlers.update(dict.fromkeys(LOOKING_TOOLS, inspect))
    return handlers


def _refused(
    call: ChatCompletionMessageToolCall,
    name: str,
    detail: str,
    response: ToolResponse,
    *,
    tag: str = 'refused',
) -> ExtendedToolResponse:
    return ExtendedToolResponse(
        tool_call_id=call_id(call), name=name, content=f'<{tag}>\n{detail}\n</{tag}>', response=response
    )


class AgentWithUserInput(InstalledNodeCliInput):
    """Inherited: user_prompt, working_dir, timeout, npm_registry.

    ``working_dir`` stays required. A task that keeps it out of ``run_agent``'s reach
    passes it through ``agent_config``.
    """

    requirements: list[str] = Field(default_factory=list)
    """What the user wants, one entry per thing it will ask for, in order.

    Wins over the inherited ``user_prompt`` whenever it is non-empty.
    """

    persona: str = 'default'
    user_client_name: str = 'main'
    worker_client_name: str = 'main'
    worker_backend: str = DEFAULT_BACKEND
    worker_config: dict[str, Any] = Field(default_factory=dict)
    skill_dirs: list[str] = Field(default_factory=list)
    persona_dirs: list[str] = Field(default_factory=list)
    """Directories holding persona files, read before ``persona`` is resolved."""
    hooks: list[dict[str, Any]] = Field(default_factory=list)
    """The topic's own logic, each as ``{'name': ..., 'args': {...}}``. See ``user_agent.hooks``."""
    extra_tools: list[str] = Field(default_factory=list)
    """Tools this run holds on top of its persona's, by ``toolset.CATALOGUE`` name."""
    max_turns: int = queue_mod.DEFAULT_TURN_CAP
    """How many turns the user gets before the run ends unfinished."""
    context_window: int = 0
    """The user model's context size, in tokens. 0 leaves the history uncompacted."""
    compact_pct: int = DEFAULT_COMPACT_PCT
    """Share of ``context_window`` the history may fill before it is summarised."""
    compact_prompt: str = SUMMARY_PROMPT
    """What to ask for when summarising the user's own history."""
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    """MCP servers the user simulator may call, as ``load_mcp_tools`` takes them."""
    reminder_light: str = LIGHT_TEXT
    """The one-line persona reminder. Blank stays quiet."""
    reminder_heavy_every: int = HEAVY_EVERY
    """Turns between full persona reminders."""
    reminder_light_every: int = LIGHT_EVERY
    """Turns without any reminder before the light one fires."""
    rate_experience: bool = True
    """Whether to ask the user to rate the collaboration once the run has ended."""


class AgentWithUserOutput(InstalledAgentOutput):
    """The installed-agent shape, plus what the conversation did.

    ``output`` is the user's verdict on the work. The counters a caller reads per run --
    every turn, idle turn, message and compaction -- live in :attr:`run_meta` rather than
    beside these fields; only ``turns`` and ``worker_rounds`` are also here, because
    recipes read them directly.
    """

    turns: int = 0
    worker_rounds: int = 0
    accepted: list[dict[str, str]] = Field(default_factory=list)
    """Each acceptance as the user worded it, in order."""
    failures: int = 0
    reached_max_turns: bool = False
    session_lost: bool = False
    """Whether a turn tore the worker's session down, which ends the run."""
    experience: ExperienceRating = Field(default_factory=ExperienceRating)
    """What the user thought of the worker. Read ``experience.rated`` before the scores."""
    usage_summary: dict[str, Any] = Field(default_factory=dict)
    """Token usage per trace label, from the run-level usage processor."""
    trajectories: dict[str, Any] = Field(default_factory=dict, exclude=True)
    """The worker's conversations and the user's, keyed by side. In-memory only.

    Kept out of ``model_dump`` because it is the run-level ``chat_tracker`` snapshot
    resliced by label -- the same content the run record already carries under
    ``processor_results.chat``, and the largest part of it. Readers of an exported run
    take it from there (``labels.group_by_label`` buckets it by side); readers holding
    this object, such as the reward-hacking detector, use this field.
    """
    run_meta: dict[str, Any] = Field(default_factory=dict)
    """How the run was configured and what it did: persona, backend, hooks, counters."""
    hook_results: dict[str, Any] = Field(default_factory=dict)
    """What the hooks asked to record, under the keys they chose."""
    hook_failures: list[str] = Field(default_factory=list)
    """One entry per event a hook raised on, naming the event."""


@agent_registry.register(AGENT_NAME)
async def run_agent_with_user(agent_input: AgentWithUserInput) -> AgentWithUserOutput:
    """Run one worker behind a simulated user, and report what happened.

    Raises for a configuration with nothing to ask for, and for a hook that cannot be
    resolved. Everything after that is reported rather than raised.
    """
    started = time.monotonic()
    queue = queue_mod.from_task_input(
        requirements=agent_input.requirements,
        user_prompt=agent_input.user_prompt,
    )
    hooks = resolve_hooks(agent_input.hooks, events=EVENTS)
    hook_names = [str(config.get('name') or '') for config in agent_input.hooks]

    state = RunState(queue=queue, natural_language=detect('\n'.join(queue.segments)))
    logger.info('user writes in %s', state.natural_language.name)
    try:
        context = require_context()
        labels.require_labelling_tracker()
        load_persona_dirs(agent_input.persona_dirs)
        persona = get_persona(agent_input.persona).for_language(state.natural_language.code)
        skills = skills_mod.load_user_skills(external=agent_input.skill_dirs)
        # Loaded before the tool set is built, which captures its schemas once.
        if agent_input.mcp_servers:
            loaded = await context.load_mcp_tools(extra_servers=agent_input.mcp_servers)
            logger.info('mcp servers=%d tools=%s', len(agent_input.mcp_servers), sorted(loaded))
        tools = toolset.build(
            toolset.tool_names(persona.tools, skills, agent_input.extra_tools),
            skills=skills,
        )

        state.manager = make_worker_manager(
            agent_input.worker_backend,
            default_working_dir=agent_input.working_dir,
            default_config=agent_input.worker_config,
            openai_client_name=agent_input.worker_client_name,
        )
        await state.manager.setup()
        state.session = await state.manager.start(name=WORKER_NAME, workspace=agent_input.working_dir)
        await skills_mod.install_skills(await context.get_runtime(), skills)

        await notify_hooks(hooks, RunStart(state=state), state)

        built = build_handlers(state, hooks=hooks)
        handlers = {name: handler for name, handler in built.items() if name in tools.schema_names}
        check_handlers(handlers, set(tools.schema_names))

        # Gives the client the _chat_phase attribute every turn below exports under.
        client = labels.enable_labels(context.get_openai_client(agent_input.user_client_name))
        history = [
            {
                'role': 'system',
                'content': build_system_prompt(
                    persona,
                    usage_docs=(*tools.usage_docs, ('worker_report', WORKER_REPORT_PROTOCOL)),
                    environment=environment_facts(
                        working_dir=agent_input.working_dir,
                        workers=[WORKER_NAME],
                        product=state.manager.product,
                    ),
                    worker_skills=worker_skills_section(state.manager.commands, prefix=state.manager.command_prefix),
                    user_memory=user_memory_section(
                        memory_text(agent_input.worker_config) if persona.reads_memory else ''
                    ),
                    may_edit=tools.may_edit,
                ),
            },
            {
                'role': 'user',
                'content': build_opening_message(
                    queue.current,
                    language=state.natural_language.name,
                    catalogue=catalogue(skills),
                ),
            },
        ]
        dispatcher = Dispatcher(executor=tools.executor, handlers=handlers)
        cap = agent_input.max_turns
        logger.info('user simulator assembled: persona=%s %s', persona.name, tools.describe())

        context.tool_state.update(toolset.tool_state(tools, working_dir=agent_input.working_dir))

        # With no recalls declared, every reminder is the light one.
        reminders = Reminders(
            recalls=persona.recalls,
            light=agent_input.reminder_light,
            heavy_every=agent_input.reminder_heavy_every,
            light_every=agent_input.reminder_light_every,
            catalogue=catalogue(skills),
            worker_commands=worker_command_lines(state.manager.commands, prefix=state.manager.command_prefix),
        )
        budget = Budget(window=agent_input.context_window, pct=agent_input.compact_pct)
        if budget.enabled:
            logger.info('user history compacts above %d tokens', budget.threshold)

        with labels.labelled(client, USER_LABEL):
            # A turn that only produced prose does not spend one of the user's turns.
            # The outer bound counts every request, including those.
            for _ in range(cap * 2):
                if state.turns >= cap:
                    break
                response = await client.create_completion(messages=history, tools=list(tools.schemas))
                message, calls = read_response(response)
                history.append(message)

                if not calls:
                    state.idle_turns += 1
                    history.append({'role': 'user', 'content': reminder(NUDGE)})
                    continue

                state.turns += 1
                # Every call is answered before anything else joins the history.
                appended: list[dict[str, Any]] = []
                for call in calls:
                    result = await dispatcher.dispatch(call)
                    history.append(tool_message(call, result.content))
                    appended.extend(result.messages)
                    state.failures += int(result.failed)
                    recall = reminders.after_tool(call_name(call))
                    if recall:
                        appended.append(reminder_message(recall))
                history.extend(appended)

                if state.stopped:
                    break

                # At a turn boundary, where every tool call has its answer.
                if budget.exceeded(response):
                    history = await compact(
                        history,
                        client=client,
                        queue=queue,
                        language=state.natural_language.name,
                        prompt=agent_input.compact_prompt,
                    )
                    state.compactions += 1
                    reminders.compacted()

            # Inside the label. Only a run the user ended itself is rated.
            if agent_input.rate_experience and state.stopped and not state.session_lost:
                state.experience = await rate_experience(history, client=client)
                logger.info(
                    'experience rating: rated=%s attempts=%d', state.experience.rated, state.experience.attempts
                )
    except Exception as exc:  # noqa: BLE001 - a failed run still reports what it managed
        state.error = f'{type(exc).__name__}: {exc}'
        logger.exception('agent_with_user failed')
    finally:
        usage = _usage_summary()
        if state.manager is not None:
            try:
                # Both run before shutdown, which drops the sessions the traces key on.
                await notify_hooks(hooks, RunFinish(state=state), state)
            except RequiredHookError as exc:
                # The loop has already ended, so this reaches the output rather than it.
                state.error = state.error or str(exc)
            trajectories = _collect_traces(state.manager, state.session)
            try:
                await state.manager.shutdown()
            except Exception as exc:  # noqa: BLE001 - shutdown must not mask the outcome
                logger.warning('worker shutdown failed: %s', exc)
        else:
            trajectories = {}

    return AgentWithUserOutput(
        success=state.succeeded,
        output=accepted_text(state.accepted),
        stderr=state.error,
        exit_code=1 if state.error else 0,
        execution_time=time.monotonic() - started,
        turns=state.turns,
        worker_rounds=state.worker_rounds,
        accepted=list(state.accepted),
        failures=state.failures,
        reached_max_turns=not state.stopped and not state.error,
        session_lost=state.session_lost,
        experience=state.experience,
        usage_summary=usage,
        trajectories=trajectories,
        run_meta={
            'persona': agent_input.persona,
            'natural_language': state.natural_language.code,
            'worker_backend': agent_input.worker_backend,
            'hooks': hook_names,
            'status': state.status,
            'stop_reason': state.stop_reason,
            'completed_steps': len(state.accepted),
            'total_steps': queue.total,
            'step_summaries': [entry.get('summary', '') for entry in state.accepted],
            'turns': state.turns,
            'idle_turns': state.idle_turns,
            'messages_sent': state.messages_sent,
            'compactions': state.compactions,
        },
        hook_results=dict(state.hook_results),
        hook_failures=list(state.hook_failures),
    )


async def notify_hooks(hooks: Sequence[RunHook], event: HookEvent, state: RunState) -> None:
    """Hand ``event`` to each hook subscribing to its class, in the order configured.

    What a hook returns lands in ``state.hook_results``, and an exception it raises in
    ``state.hook_failures``. A hook whose ``on_failure`` returns True raises
    :class:`RequiredHookError` instead, and the hooks after it do not run.
    """
    for hook in hooks:
        if type(event) not in hook.events:
            continue
        try:
            outcome = await hook.on_event(event)
        except Exception as exc:  # noqa: BLE001 - recorded on the output
            logger.exception('hook failed on %s', event.name)
            state.hook_failures.append(f'{event.name}: {type(exc).__name__}: {exc}')
            if await hook.on_failure(event, exc):
                raise RequiredHookError(event.name, exc) from exc
            continue
        if outcome is not None:
            state.hook_results.update(outcome.results)


def _usage_summary() -> dict[str, Any]:
    """Token usage for the run, keyed by trace label.

    Covers the user's own turns and its compaction summaries alongside each worker's
    requests. Empty when no usage processor is installed.
    """
    processor = get_processor(HITLUsageProcessor)
    if not isinstance(processor, HITLUsageProcessor):
        logger.warning('no usage processor installed; the run reports no token usage')
        return {}
    return dict(processor.get_result())


def _collect_traces(manager: SessionManager, session: Session | None) -> dict[str, Any]:
    """The worker's trajectories and the user's, kept apart by their labels."""
    collected: dict[str, Any] = {}
    if session is not None:
        try:
            collected['worker'] = {'traces': manager.traces(session)}
        except Exception as exc:  # noqa: BLE001 - traces are diagnostic, not the result
            logger.warning('could not collect worker traces: %s', exc)
    try:
        run_level = labels.run_level_trajectories()
        collected['user'] = {'traces': labels.filter_by_label(run_level, USER_LABEL)}
        stranded = labels.unattributed_count(run_level)
        if stranded:
            logger.warning('%d trajectories carry no label and no filter claims them', stranded)
    except Exception as exc:  # noqa: BLE001 - as above
        logger.warning('could not collect user traces: %s', exc)
    return collected


__all__ = [
    'DEFAULT_BACKEND',
    'EVENTS',
    'LOOKING_TOOLS',
    'USER_LABEL',
    'WORKER_NAME',
    'AgentWithUserInput',
    'AgentWithUserOutput',
    'RunFinish',
    'RunStart',
    'RunState',
    'WorkerDone',
    'accepted_text',
    'build_handlers',
    'notify_hooks',
    'run_agent_with_user',
]
