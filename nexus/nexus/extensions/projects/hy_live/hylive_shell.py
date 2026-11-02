"""Run one hy-live episode on hylive: the core decides, this module carries out its commands in nexus.

hylive's ``step`` plays the cerebellum and the midbrain side. Around it, this shell:

- drives the simulated user against the cerebellum's speech, as the episode loop always did;
- answers each command through a port: model calls through the ctx clients, ``web_search`` through the
  search backend, ACP primitives through ``AcpSessionExecutor`` on the v1 boundary, Task directories and
  their files through the sandbox, and ACP request answers through ``AcpRequestRelay``;
- turns the core's state into the episode's outputs, and records and traces what it exchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
import uuid
import weakref

from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, cast

from hylive import commands, events
from hylive.harness import WORKING_RUN, Harness
from hylive.midbrain_loop import CONCLUDE_SOURCE
from hylive.notices import receipt_status
from hylive.runner import SessionRunner
from hylive.state import SessionConfig, SessionState
from hylive.step import cerebellum_idle, has_active_runs, has_handle_jobs, has_pending_notices, new_session

from nexus.core import ExecutionContext
from nexus.core.openai_client import ContextLengthExceededError
from nexus.extensions.agents.base.schema import ToolChoice
from nexus.extensions.agents.installed_common.host_rpc import (
    ACP_ELICITATION_REQUEST_EVENT,
    ACP_ELICITATION_RESOLVE_EVENT,
    ACP_PERMISSION_REQUEST_EVENT,
    ACP_PERMISSION_RESOLVE_EVENT,
)
from nexus.trace import trace_attributes_context

from .coordination.acp_updates import AcpUpdateRouter
from .coordination.backend import PROMPT_BUDGET_S, AcpSessionExecutor, WorkAgentBudget
from .coordination.models import AgentTask, OwnerEvent, SessionRef, TaskError, TaskResult, TaskStatus
from .episode_clock import EpisodeClock
from .health import FORMAT_ERROR_TERMINATION, TRUNCATION_TERMINATION, USER_LEFT_EARLY_TERMINATION
from .recording import EventLog
from .trace_spans import CEREBELLUM_LAYER, MIDBRAIN_LAYER, midbrain_request_attributes, new_request_id, record_tool_call
from .user_simulator import (
    CerebellumResponse,
    SimulatedConversation,
    SimulatedTurn,
    UserMessage,
    UserSimulator,
    UserVisibleConversation,
)
from .web_search import WebSearchBackend


logger = logging.getLogger(__name__)

_SANDBOX_ARTIFACT_LIST_COMMAND = (
    "find . \\( -path '*/.*' -o -name node_modules -o -name __pycache__ -o -name venv -o -name env \\) "
    '-prune -o -type f -print'
)
_SANDBOX_PROJECT_PREPARE_COMMAND = 'mkdir -p -- "$HY_LIVE_PROJECT_DIR"'
_PERMISSION_FACT = '有一条权限请求需要用户决定'
# NOTE(wallyshi): Episode end waits this long for the midbrain and the sessions to settle; a model call
# that hangs must not keep the episode, and its sandbox, open forever.
_CLOSE_DRAIN_TIMEOUT_S = 60.0
_TOOL_CALL_MARKUP = re.compile(r'<[｜|]+DSML[｜|]+|</?tool_calls?(?::[^\s>]+)?>')
"""A tool call left as text: DeepSeek DSML when the server skips its tool parser (``tool_choice=none``), and
HY4 structural tokens when the HyTuner parser falls back to raw text on a malformed call."""
MARKUP_PROBLEM = 'Cerebellum emitted tool-call markup instead of structured tool calls'
_DEADLINE_POLL_S = 5.0
"""How often a wait re-reads the delivery deadline, which a frozen episode pushes out."""


@dataclass
class Episode:
    conversation: SimulatedConversation
    summary: str
    tasks: list[AgentTask]
    delivery_events: list[OwnerEvent] = field(default_factory=list)
    delivery_responses: list[CerebellumResponse] = field(default_factory=list)
    episode_termination: str | None = None
    clock: dict[str, float] = field(default_factory=dict)
    """``EpisodeClock.summary()`` at the end: wall time, time the cerebellum was paused, time frozen."""


class AcpRequestRelay:
    """Bind each turn's channel, relay its ACP permission and elicitation requests to the core, send answers back.

    The core answers a request by the key this relay gave it; the relay only remembers how to send that answer.
    """

    def __init__(self, runner: SessionRunner) -> None:
        self.runner = runner
        self._bindings: dict[str, tuple[str, str]] = {}
        self._permission_answers: dict[str, Callable[[str, list[dict[str, Any]] | None], Awaitable[None]]] = {}
        self._elicitation_answers: dict[str, Callable[[dict[str, Any]], Awaitable[None]]] = {}
        self._runtime_ref: weakref.ReferenceType[Any] | None = None
        self._requests = 0
        """Numbers the keys: a CLI restarts its request ids, and only the relay's own key tells two requests apart."""

    def bind(self, channel_id: str, *, owner_id: str, turn_id: str) -> None:
        self._bindings[channel_id] = (owner_id, turn_id)

    def unbind(self, channel_id: str) -> None:
        self._bindings.pop(channel_id, None)

    def listen(self, runtime: Any) -> None:
        """Subscribe to the runtime's ACP requests; later calls for the same runtime are no-ops."""
        if self._runtime_ref is not None and self._runtime_ref() is runtime:
            return
        self._runtime_ref = weakref.ref(runtime)
        runtime.on(ACP_PERMISSION_REQUEST_EVENT, cast(Any, self.dispatch))
        runtime.on(ACP_ELICITATION_REQUEST_EVENT, cast(Any, self.dispatch_elicitation))

    def _runtime(self) -> Any:
        return self._runtime_ref() if self._runtime_ref is not None else None

    async def dispatch(self, data: dict[str, Any]) -> None:
        channel_id = data.get('channel_id') if isinstance(data, dict) else None
        params = data.get('params') if isinstance(data, dict) else None
        token = data.get('permission_token') if isinstance(data, dict) else None
        request_id = data.get('acp_request_id') if isinstance(data, dict) else None
        if not isinstance(channel_id, str) or not isinstance(params, dict) or not isinstance(token, str) or not token:
            return
        if not isinstance(request_id, (int, str)):
            return
        try:
            acp_request_id = int(request_id)
        except ValueError:
            return
        runtime = self._runtime()
        binding = self._bindings.get(channel_id)
        if runtime is None:
            return
        self._requests += 1
        key = f'{channel_id}#{acp_request_id}#{self._requests}'

        async def answer(option_id: str, ext_requests: list[dict[str, Any]] | None) -> None:
            payload: dict[str, Any] = {
                'channel_id': channel_id,
                'acp_request_id': acp_request_id,
                'permission_token': token,
                'option_id': option_id,
            }
            if ext_requests is not None:
                payload['ext_requests'] = ext_requests
            await runtime.emit(ACP_PERMISSION_RESOLVE_EVENT, payload)

        if binding is None:
            # NOTE(wallyshi): hy-live relays permissions without a timeout, so an unanswered request would
            # hold the sandbox turn forever; one no turn owns is cancelled at once.
            await answer('', None)
            return
        self._permission_answers[key] = answer
        try:
            await self.runner.feed_and_wait(
                events.AcpPermissionRequested(at=self.runner.now(), request_key=key, turn_id=binding[1], params=params)
            )
        except Exception:
            logger.exception('Failed to hand an ACP permission to hylive channel=%s; cancelling it', channel_id)
            if self._permission_answers.pop(key, None) is not None:
                await answer('', None)

    async def dispatch_elicitation(self, data: dict[str, Any]) -> None:
        channel_id = data.get('channel_id') if isinstance(data, dict) else None
        params = data.get('params') if isinstance(data, dict) else None
        token = data.get('elicitation_token') if isinstance(data, dict) else None
        if not isinstance(channel_id, str) or not isinstance(params, dict) or not isinstance(token, str) or not token:
            return
        runtime = self._runtime()
        if runtime is None:
            return
        self._requests += 1
        key = f'{channel_id}#elicitation-{self._requests}'

        async def answer(result: dict[str, Any]) -> None:
            await runtime.emit(
                ACP_ELICITATION_RESOLVE_EVENT, {'channel_id': channel_id, 'elicitation_token': token, 'result': result}
            )

        binding = self._bindings.get(channel_id)
        if binding is None:
            await answer({'action': 'cancel'})
            return
        self._elicitation_answers[key] = answer
        try:
            await self.runner.feed_and_wait(
                events.AcpElicitationRequested(at=self.runner.now(), request_key=key, turn_id=binding[1], params=params)
            )
        except Exception:
            logger.exception('Failed to hand an ACP elicitation to hylive channel=%s; cancelling it', channel_id)
            if self._elicitation_answers.pop(key, None) is not None:
                await answer({'action': 'cancel'})

    async def respond(self, command: commands.AcpRespondPermission | commands.AcpRespondElicitation) -> None:
        if isinstance(command, commands.AcpRespondPermission):
            answer = self._permission_answers.pop(command.request_key, None)
            if answer is not None:
                await answer(command.option_id, command.ext_requests)
            return
        reply = self._elicitation_answers.pop(command.request_key, None)
        if reply is not None:
            await reply(command.result)


@dataclass
class _Driver:
    """The simulated user's side: what it has heard, and what the cerebellum is saying to it."""

    visible: UserVisibleConversation = field(default_factory=UserVisibleConversation)
    turns: list[SimulatedTurn] = field(default_factory=list)
    delivery_responses: list[CerebellumResponse] = field(default_factory=list)
    turn_speech: asyncio.Future[str] | None = None
    relays: int = 0
    """Notices the cerebellum began to relay; the episode loop treats each as a notice that arrived."""
    relaying: dict[str, Any] | None = None
    """The notice the cerebellum is relaying now."""
    relayed: int = 0
    permission_delivered: bool = False
    termination: str | None = None


class _Tracer:
    """Mark each executed tool call and each midbrain request on the trace, from what the core wrote down."""

    def __init__(self) -> None:
        self._cerebellum_seen = 0
        self._midbrain_seen = 0
        self._submitted: deque[str] = deque()
        """Request ids of the cerebellum's handed-over calls, in the order the midbrain takes them."""
        self.midbrain_request: dict[str, Any] | None = None

    def after_step(self, state: SessionState) -> None:
        messages = state.cerebellum.messages
        for index in range(self._cerebellum_seen, len(messages)):
            message = messages[index]
            if message.get('role') != 'tool':
                continue
            call = _call_for(messages, index)
            request_id = None
            if '"status": "submitted"' in str(message.get('content')):
                request_id = new_request_id()
                self._submitted.append(request_id)
            record_tool_call(
                layer=CEREBELLUM_LAYER,
                call=call,
                result=str(message.get('content') or ''),
                started_at=float(message.get('_at') or time.time()),
                request_id=request_id,
            )
        self._cerebellum_seen = len(messages)
        log = state.midbrain_log
        for index in range(self._midbrain_seen, len(log)):
            message = log[index]
            if message.get('role') != 'tool' or self.midbrain_request is None:
                continue
            with trace_attributes_context(self.midbrain_request):
                record_tool_call(
                    layer=MIDBRAIN_LAYER,
                    call=_call_for(log, index),
                    result=str(message.get('content') or ''),
                    started_at=float(message.get('_at') or time.time()),
                )
        self._midbrain_seen = len(log)

    def midbrain_call(self, state: SessionState, command: commands.CallLLM) -> dict[str, Any]:
        """The attributes of the midbrain request this call belongs to; its first call opens the request."""
        job = state.midbrain_job or {}
        if command.request_meta.get('turn_step') == 0 or self.midbrain_request is None:
            source = str(job.get('source') or command.request_meta.get('source') or '')
            if source == CONCLUDE_SOURCE:
                request_id = new_request_id()
            else:
                request_id = self._submitted.popleft() if self._submitted else new_request_id()
            self.midbrain_request = midbrain_request_attributes(
                request_id=request_id, source=source, text=str(job.get('text') or ''), path=str(job.get('path') or '')
            )
        return self.midbrain_request


def _prompt_tokens(response: Any) -> int | None:
    usage = getattr(response, 'usage', None)
    if usage is None and isinstance(response, dict):
        usage = response.get('usage')
    if usage is None:
        return None
    value = usage.get('prompt_tokens') if isinstance(usage, dict) else getattr(usage, 'prompt_tokens', None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _midbrain_context_window(client: Any) -> int | None:
    """The midbrain model's input window, from its request args or LiteLLM's catalog."""
    config = getattr(client, 'config', None)
    args = getattr(config, 'request_args', None)
    if isinstance(args, dict):
        for key in ('max_input_tokens', 'context_window'):
            value = args.get(key)
            if isinstance(value, int) and value > 0:
                return value
        model = args.get('model')
    else:
        model = None
    if not isinstance(model, str) or not model:
        return None
    try:
        from hylive.midbrain_loop import model_context_window
    except ImportError:
        return None
    return model_context_window(model)


def _dump_message(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    if hasattr(message, 'model_dump'):
        return message.model_dump(exclude_none=True)
    raise TypeError(f'A model response message must be a mapping or model, got {type(message).__name__}')


def _call_for(messages: list[dict[str, Any]], index: int) -> dict[str, Any]:
    """The tool call a tool result at ``index`` answers."""
    call_id = messages[index].get('tool_call_id')
    for message in reversed(messages[:index]):
        for call in message.get('tool_calls') or []:
            if isinstance(call, dict) and call.get('id') == call_id:
                return call
    return {'id': call_id, 'function': {'name': messages[index].get('name') or ''}}


class _NexusPorts:
    def __init__(
        self,
        *,
        ctx: ExecutionContext,
        clients: dict[str, Any],
        executor: AcpSessionExecutor,
        relay: AcpRequestRelay,
        search_backend: WebSearchBackend | None,
        driver: _Driver,
        runner: SessionRunner,
        clock: EpisodeClock,
        tracer: _Tracer,
    ) -> None:
        self.ctx = ctx
        self.clients = clients
        self.executor = executor
        self.relay = relay
        self.search_backend = search_backend
        self.driver = driver
        self.runner = runner
        self.clock = clock
        self.tracer = tracer
        self._runtime_lock = asyncio.Lock()

    async def call_llm(self, command: commands.CallLLM) -> events.LLMResult:
        client = self.clients[command.role]
        kwargs = {
            'messages': command.messages,
            'tools': command.tools,
            'tool_choice': ToolChoice(command.tool_choice),
            'parallel_tool_calls': command.parallel_tool_calls,
            'request_meta': command.request_meta,
        }
        try:
            if command.role == 'cerebellum':
                with self.clock.cerebellum_call():
                    response = await client.create_completion(**kwargs)
            else:
                with trace_attributes_context(
                    self.tracer.midbrain_call(cast(SessionState, self.runner.state), command)
                ):
                    response = await client.create_completion(**kwargs)
        except ContextLengthExceededError as error:
            return events.LLMResult(
                at=self.runner.now(), call_id=command.call_id, error=str(error), error_kind='context_length'
            )
        message = _dump_message(response.choices[0].message)
        if command.role == 'cerebellum' and _TOOL_CALL_MARKUP.search(str(message.get('content') or '')):
            logger.warning('Cerebellum answered with tool-call markup: %.500s', message.get('content'))
            return events.LLMResult(
                at=self.runner.now(), call_id=command.call_id, error=MARKUP_PROBLEM, error_kind='format'
            )
        return events.LLMResult(
            at=self.runner.now(), call_id=command.call_id, message=message, prompt_tokens=_prompt_tokens(response)
        )

    async def web_search(self, command: commands.WebSearch) -> str:
        assert self.search_backend is not None
        return await self.search_backend.search(command.query, prompt=command.prompt)

    async def backend(self, command: commands.Command) -> Any:
        if isinstance(command, commands.PrepareProject):
            await self._prepare_project(command.working_dir)
            return None
        session = SessionRef(**command.session.model_dump())  # type: ignore[union-attr]
        if isinstance(command, commands.AcpOpenSession):
            await self.executor.open_session(session)
        elif isinstance(command, commands.AcpPrompt):
            await self.executor.prompt(session, turn_id=command.turn_id, prompt=command.prompt)
        elif isinstance(command, commands.AcpInterrupt):
            await self.executor.interrupt(session)
        elif isinstance(command, commands.AcpSteerTurn):
            return await self.executor.steer(session, prompt=command.prompt)
        elif isinstance(command, commands.AcpSideQuestion):
            return await self.executor.side_question(session, question=command.question)
        elif isinstance(command, commands.AcpCloseSession):
            await self.executor.close_session(session)
        else:
            raise ValueError(f'The v1 nexus shell does not run {type(command).__name__}')
        return None

    async def list_artifacts(self, command: commands.ListArtifacts) -> list[str]:
        runtime = await self._runtime()
        response = await runtime.run_command(_SANDBOX_ARTIFACT_LIST_COMMAND, cwd=command.working_dir, timeout=30)
        if response.return_code != 0:
            raise RuntimeError(
                f'Failed to list sandbox artifacts under {command.working_dir}: {response.stderr or response.status}'
            )
        root = PurePosixPath(command.working_dir)
        return [str(root / line.strip()) for line in (response.stdout or '').splitlines() if line.strip()]

    async def respond(self, command: commands.AcpRespondPermission | commands.AcpRespondElicitation) -> None:
        await self.relay.respond(command)

    def speak(self, command: commands.Speak) -> None:
        driver = self.driver
        if command.source == 'user':
            if driver.turn_speech is not None and not driver.turn_speech.done():
                driver.turn_speech.set_result(command.text)
            return
        notice = driver.relaying or {}
        driver.relaying = None
        driver.relayed += 1
        driver.permission_delivered = driver.permission_delivered or _PERMISSION_FACT in str(notice.get('fact') or '')
        response = CerebellumResponse(
            content=command.text,
            metadata={
                'source': 'notification',
                'notice_kinds': list(notice.get('kinds') or [notice.get('kind')]),
                'receipt_status': receipt_status(notice) if notice else None,
            },
        )
        driver.visible.record_cerebellum(response)
        driver.delivery_responses.append(response)

    def failed(self, command: commands.SessionFailed) -> None:
        logger.warning('hy-live episode ended by the core: %s %s', command.reason, command.detail)
        self.driver.termination = {
            'FORMAT_ERROR': FORMAT_ERROR_TERMINATION,
            'TRUNCATION': TRUNCATION_TERMINATION,
        }.get(command.reason, command.reason)
        if self.driver.turn_speech is not None and not self.driver.turn_speech.done():
            self.driver.turn_speech.set_result('')

    async def _runtime(self) -> Any:
        # NOTE(jaxlliu): ExecutionContext.get_runtime caches a runtime only once its acquisition finishes, so
        # two Tasks starting while the sandbox comes up would each acquire it and hold separate clients of one
        # sandbox. Acquiring one at a time leaves a single client for every later caller.
        async with self._runtime_lock:
            return await self.ctx.get_runtime()

    async def _prepare_project(self, working_dir: str) -> None:
        runtime = await self._runtime()
        response = await runtime.run_command(
            _SANDBOX_PROJECT_PREPARE_COMMAND, timeout=30, extra_env={'HY_LIVE_PROJECT_DIR': working_dir}
        )
        if response.return_code != 0:
            raise RuntimeError(
                f'Failed to prepare sandbox project directory {working_dir}: {response.stderr or response.status}'
            )


@dataclass
class LiveEpisode:
    episode: Episode
    state: SessionState


class LiveShell:
    """One owner's hy-live conversation on hylive: the runner, its ports and the backend executor."""

    def __init__(
        self,
        *,
        ctx: ExecutionContext,
        owner_id: str,
        clients: dict[str, Any],
        cerebellum_system: str,
        trusted_config: dict[str, Any],
        budget: WorkAgentBudget,
        search_backend: WebSearchBackend | None,
        clock: EpisodeClock,
        session_cache_task_id: bool = False,
        event_log: EventLog | None = None,
        summarize_outcomes: bool = False,
    ) -> None:
        self.owner_id = owner_id
        self.clock = clock
        self.event_log = event_log
        workbuddy = trusted_config.get('acp_cli') == 'workbuddy'
        self.config = SessionConfig(
            owner_id=owner_id,
            cerebellum_system=cerebellum_system,
            project_working_dir=trusted_config.get('working_dir', '/workspace'),
            id_seed=uuid.uuid4().hex,
            search_enabled=search_backend is not None,
            registers_artifacts_from_acp=workbuddy,
            supports_steer=workbuddy,
            supports_side_question=workbuddy,
            boundary='v1',
            turn_budget_s=PROMPT_BUDGET_S,
            wall_timeout_s=budget.timeout_s,
            summarize_outcomes=summarize_outcomes,
            midbrain_context_window=_midbrain_context_window(clients.get('midbrain')),
        )
        self.driver = _Driver()
        self.tracer = _Tracer()
        self._started = False
        self.runner = SessionRunner(
            new_session(self.config, at=time.time()),
            cast(Any, None),
            observer=self._observe,
            suspended=lambda: clock.frozen_s,
        )
        self.relay = AcpRequestRelay(self.runner)
        runner = self.runner

        def update_sink(_owner_id: str, turn_id: str, update: dict[str, Any]) -> None:
            runner.feed(events.AcpUpdate(at=runner.now(), turn_id=turn_id, update=dict(update)))

        async def turn_ended_sink(turn_id: str, session_id: str, result: TaskResult) -> None:
            await runner.feed_and_wait(
                events.AcpTurnEnded(
                    at=runner.now(), turn_id=turn_id, session_id=session_id, result=result.model_dump(mode='json')
                )
            )

        def owner_of(turn_id: str) -> str:
            turn = cast(SessionState, runner.state).midbrain.turns.get(turn_id)
            return str(turn['owner']) if turn is not None else turn_id

        self.executor = AcpSessionExecutor(
            ctx=ctx,
            update_router=AcpUpdateRouter(update_sink=update_sink),
            turn_ended_sink=turn_ended_sink,
            request_binder=self.relay,
            trusted_config=trusted_config,
            budget=budget,
            session_cache_task_id=session_cache_task_id,
            owner_of=owner_of,
        )
        self.runner.ports = _NexusPorts(
            ctx=ctx,
            clients=clients,
            executor=self.executor,
            relay=self.relay,
            search_backend=search_backend,
            driver=self.driver,
            runner=self.runner,
            clock=clock,
            tracer=self.tracer,
        )
        if event_log is not None:
            event_log.config(self.config.model_dump(mode='json'))

    @property
    def state(self) -> SessionState:
        return cast(SessionState, self.runner.state)

    def background_working(self) -> bool:
        """Whether anything besides the cerebellum is making progress: a Run working, or the midbrain on a call.

        A Run that waits for a permission or an answer is not working: its next step needs the cerebellum.
        """
        state = self.state
        if state.midbrain_job is not None or state.midbrain_jobs:
            return True
        return any(task.run is not None and task.run.status in WORKING_RUN for task in state.harness.tasks.values())

    async def warm_up(self) -> None:
        """Bring the sandbox up before the conversation starts; until then no Task can do any work."""
        await self.executor.warm_up()

    def _observe(self, direction: str, item: Any) -> None:
        if self.event_log is not None:
            if direction == 'in':
                self.event_log.event(item)
            else:
                self.event_log.command(item)
        if direction == 'out' and isinstance(item, commands.CallLLM) and item.role == 'cerebellum':
            purpose = self.state.cerebellum.purpose
            notice = purpose.get('notice') if purpose is not None else None
            if item.request_meta.get('source') == 'observe' and notice and self.driver.relaying is not notice:
                self.driver.relaying = notice
                self.driver.relays += 1
        if direction == 'out' and isinstance(item, (commands.Speak, commands.CallLLM, commands.SessionFailed)):
            self.tracer.after_step(self.state)

    async def run(
        self,
        *,
        simulator: UserSimulator,
        max_user_turns: int,
        task_delivery_timeout: float,
        hold_next_user_until_permission: bool = False,
    ) -> LiveEpisode:
        """Race simulated user turns with background notices until all accepted work settles.

        ``task_delivery_timeout`` is measured on the episode clock, which leaves out the time the episode was
        frozen on a paused cerebellum.
        """
        self.clock.start()
        self.runner.start()
        self._started = True
        try:
            episode = await self._drive(
                simulator=simulator,
                max_user_turns=max_user_turns,
                task_delivery_timeout=task_delivery_timeout,
                hold_next_user_until_permission=hold_next_user_until_permission,
            )
        finally:
            await self.clock.stop()
        episode.clock = self.clock.summary()
        logger.info(
            'hy-live episode clock owner=%s wall_s=%.1f cerebellum_paused_s=%.1f frozen_s=%.1f termination=%s',
            self.owner_id,
            episode.clock['wall_s'],
            episode.clock['cerebellum_paused_s'],
            episode.clock['frozen_s'],
            episode.episode_termination,
        )
        return LiveEpisode(episode=episode, state=self.state)

    async def close(self) -> None:
        """Let the midbrain finish, close every project session, then stop the runner and the executor."""
        runner = self.runner
        if not self._started:
            await runner.close()
            await self.executor.aclose()
            return
        try:
            runner.feed(events.SessionClosing(at=runner.now()))
            await runner.until(lambda current: current.midbrain.closed, timeout=_CLOSE_DRAIN_TIMEOUT_S)
            await runner.until(_sessions_closed, timeout=_CLOSE_DRAIN_TIMEOUT_S)
        finally:
            await runner.close()
            await self.executor.aclose()

    async def _drive(
        self,
        *,
        simulator: UserSimulator,
        max_user_turns: int,
        task_delivery_timeout: float,
        hold_next_user_until_permission: bool,
    ) -> Episode:
        runner = self.runner
        driver = self.driver
        visible = driver.visible
        ended_by_user = False
        reached_max_turns = False
        next_turn_gate_released = False

        def deadline_remaining() -> float:
            return task_delivery_timeout - self.clock.elapsed()

        def ended() -> bool:
            return driver.termination is not None

        async def until(predicate: Callable[[SessionState], bool], *, deadline: bool = False) -> bool:
            """Wait for ``predicate`` after a step; with ``deadline``, False once the delivery deadline passed."""
            while True:
                timeout = min(deadline_remaining(), _DEADLINE_POLL_S) if deadline else None
                if timeout is not None and timeout <= 0:
                    return predicate(cast(SessionState, runner.state))
                if await runner.until(predicate, timeout=timeout):
                    return True

        def _relay_after(seen: int) -> Callable[[SessionState], bool]:
            return lambda _: driver.relays > seen or ended()

        async def next_relay(*, deadline: bool) -> bool:
            """Wait for the cerebellum to start relaying a notice, then for it to finish; False at the deadline."""
            if not await until(_relay_after(driver.relays), deadline=deadline):
                return False
            await until(lambda _: driver.relayed >= driver.relays or ended())
            return not ended()

        async def run_user_turn(message: UserMessage) -> None:
            if not simulator.is_message_current(message, visible):
                return
            visible.record_user(message)
            driver.turn_speech = asyncio.get_running_loop().create_future()
            runner.feed(events.UserMessage(at=runner.now(), content=message))
            speech = await driver.turn_speech
            if ended():
                return
            response = CerebellumResponse(content=speech, metadata={'source': 'user'})
            visible.record_cerebellum(response)
            driver.turns.append(
                SimulatedTurn(index=len(driver.turns), user_message=message, cerebellum_response=response)
            )
            await until(lambda current: not has_handle_jobs(current) or ended())

        while len(driver.turns) < max_user_turns and not ended():
            gate = simulator.next_turn_gate()
            must_wait_for_notice = not next_turn_gate_released and (
                gate in {'after_delivery', 'after_progress'}
                or (
                    hold_next_user_until_permission
                    and not driver.permission_delivered
                    and bool(driver.turns)
                    and has_active_runs(self.state)
                )
            )
            if must_wait_for_notice and not has_active_runs(self.state) and not has_pending_notices(self.state):
                # NOTE(wallyshi): With no work running and nothing queued, no notice can ever arrive; waiting
                # would only burn the delivery deadline, so the gated turn speaks now.
                must_wait_for_notice = False
            if must_wait_for_notice:
                if not await next_relay(deadline=True):
                    break
                next_turn_gate_released = True
                continue

            user_task = asyncio.create_task(simulator.next_user_message(visible), name='hy-live-user')
            relay_task = asyncio.create_task(until(_relay_after(driver.relays)), name='hy-live-relay')
            done, _ = await asyncio.wait({user_task, relay_task}, return_when=asyncio.FIRST_COMPLETED)
            if relay_task in done:
                await until(lambda _: driver.relayed >= driver.relays or ended())
                if ended():
                    user_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await user_task
                    break
                if user_task.done():
                    message = user_task.result()
                    if message is None:
                        if not simulator.is_message_current(message, visible):
                            continue
                        ended_by_user = True
                        break
                    await run_user_turn(message)
                    next_turn_gate_released = False
                else:
                    user_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await user_task
                continue

            message = user_task.result()
            relay_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await relay_task
            if message is None:
                if not simulator.is_message_current(message, visible):
                    continue
                ended_by_user = True
                break
            await run_user_turn(message)
            next_turn_gate_released = False
        else:
            reached_max_turns = not ended()

        if ended():
            return self._episode(SimulatedConversation(), termination=driver.termination)
        if ended_by_user or reached_max_turns:
            runner.feed(events.UserFinished(at=runner.now()))
        await until(
            lambda current: (
                ended()
                or (
                    not has_active_runs(current)
                    and not has_pending_notices(current)
                    and not has_handle_jobs(current)
                    and cerebellum_idle(current)
                    and driver.relayed >= driver.relays
                )
            ),
            deadline=True,
        )
        if ended():
            return self._episode(SimulatedConversation(), termination=driver.termination)
        state = self.state
        if state.infrastructure_failures:
            termination: str | None = 'ENV_ERROR'
        elif has_active_runs(state):
            termination = 'TASK_TIMEOUT'
        elif state.declined_for_absent_user:
            termination = USER_LEFT_EARLY_TERMINATION
        else:
            termination = None
        return self._episode(
            SimulatedConversation(turns=driver.turns, ended_by_user=ended_by_user, reached_max_turns=reached_max_turns),
            termination=termination,
        )

    def _episode(self, conversation: SimulatedConversation, *, termination: str | None) -> Episode:
        state = self.state
        visible = self.driver.visible
        failed = termination in {FORMAT_ERROR_TERMINATION, TRUNCATION_TERMINATION}
        return Episode(
            conversation=conversation,
            summary='' if failed else (visible.latest_response.content if visible.latest_response is not None else ''),
            tasks=harness_agent_tasks(state, self.owner_id),
            delivery_events=[] if failed else notice_owner_events(state.notice_log, state.harness, self.owner_id),
            delivery_responses=[] if failed else self.driver.delivery_responses,
            episode_termination=termination,
        )


def _sessions_closed(state: SessionState) -> bool:
    """No session close, nor the interrupt of a turn a close found running, is still in flight."""
    return not any(item.get('kind') in {'close', 'ts_closed'} for item in state.midbrain.pending.values())


def harness_agent_tasks(state: SessionState, owner_id: str) -> list[AgentTask]:
    harness = state.harness
    tasks: list[AgentTask] = []
    for sequence, task in enumerate(harness.tasks.values(), 1):
        latest = task.run
        recorded = state.results_by_run.get(latest.path) if latest is not None else None
        result = (
            (TaskResult.model_validate(recorded) if recorded is not None else None)
            or _task_result(harness, latest.path, latest.status)
            if latest is not None
            else None
        )
        events_of_run = harness.events_for(latest.path) if latest is not None else []
        tasks.append(
            AgentTask(
                task_id=task.id,
                owner_id=owner_id,
                objective=task.objective,
                status=_agent_task_status(task.status),
                created_sequence=sequence,
                coordinator_session_id=f'hy-live:{owner_id}:midbrain',
                target_session_id=f'hy-live:{owner_id}:project:{task.id}',
                result=result,
                error=result.error if result is not None and result.execution_status == 'failed' else None,
                activity=[
                    {'id': event.id, 'kind': event.kind, 'status': event.status, 'detail': event.text}
                    for event in events_of_run[-20:]
                ],
                agent_message=events_of_run[-1].text if events_of_run else None,
            )
        )
    return tasks


def notice_owner_events(notices: list[dict[str, Any]], harness: Harness, owner_id: str) -> list[OwnerEvent]:
    owner_events: list[OwnerEvent] = []
    for sequence, notice in enumerate(notices, 1):
        run = harness.run_by_path(notice['path'])
        task_id = None if run is None else run.task_id
        if notice['kind'] == 'permission':
            kind = 'permission_request'
        elif notice['kind'] == 'input':
            kind = 'backend_input_request'
        elif notice['kind'] in {'completed', 'failed', 'cancelled', 'answer'}:
            kind = 'task_delivery'
        else:
            continue
        owner_events.append(
            OwnerEvent(
                sequence=sequence,
                owner_id=owner_id,
                kind=cast(Any, kind),
                task_id=task_id,
                payload={'content': notice['content']},
            )
        )
    return owner_events


_AGENT_TASK_STATUS: dict[str, TaskStatus] = {
    'initializing': 'queued',
    'running': 'running',
    'needs_permission': 'running',
    'needs_input': 'running',
    'completed': 'done',
    'failed': 'failed',
    'cancelled': 'cancelled',
}


def _agent_task_status(status: str) -> TaskStatus:
    return _AGENT_TASK_STATUS.get(status, 'failed')


def _task_result(harness: Harness, run_path: str, status: str) -> TaskResult | None:
    if status not in {'completed', 'failed', 'cancelled'}:
        return None
    run_events = harness.events_for(run_path)
    result_text = next((event.text for event in reversed(run_events) if event.kind in {'result', 'failed'}), '')
    if status == 'completed':
        return TaskResult(final_output=result_text)
    execution_status = 'cancelled' if status == 'cancelled' else 'failed'
    failure_kind = 'cancel' if status == 'cancelled' else 'execution'
    error = TaskError(
        code='HARNESS_CANCELLED' if status == 'cancelled' else 'HARNESS_FAILED',
        message=result_text or ('执行已取消' if status == 'cancelled' else '执行失败'),
    )
    return TaskResult(
        execution_status=execution_status, failure_kind=failure_kind, error=error, final_output=result_text
    )


__all__ = [
    'AcpRequestRelay',
    'Episode',
    'LiveEpisode',
    'LiveShell',
    'harness_agent_tasks',
    'notice_owner_events',
]
