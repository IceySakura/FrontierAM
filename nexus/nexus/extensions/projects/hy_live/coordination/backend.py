"""ACP primitives on CodeBuddy and WorkBuddy project sessions: hylive's v1 boundary.

hylive decides everything about turns: when one starts, queues, steers, times out or is interrupted. This
executor only carries out one primitive at a time on a runtime-local ACP session (``acp_cli`` in the
trusted config): open a session, run one turn, interrupt it, steer it, ask a btw question beside it, close
it. What stays here is mechanism: the Node proxy preload, each session's LLM channel and ``cache_task_id``,
binding the turn's pushed updates and ACP requests to it, and reading the turn's result.

WorkBuddy speaks ACP version 2: a steer inserts a request into the running turn, and a btw
``_codebuddy.ai/sideQuestion`` reads the session's own context and answers without tools.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, Field

from nexus.core import AgentRunOutput, ExecutionContext
from nexus.extensions.agents.codebuddy.agent import CodeBuddyMultiturnInput, run_codebuddy_multiturn
from nexus.extensions.agents.codebuddy.session_runtime import (
    AcpCli,
    close_codebuddy_session,
    interrupt_codebuddy_session,
    side_question_codebuddy_session,
    steer_codebuddy_session,
)
from nexus.extensions.agents.installed_common import call_runtime_agent, unbind_llm_channel
from nexus.extensions.agents.installed_common.multiturn import normalize_stream_json_session_id
from nexus.runtime.models import InvokeRequest
from nexus.trace import processor_context, trace_attributes_context
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor

from .acp_updates import AcpUpdateRouter
from .models import SessionRef, TaskError, TaskExecutionStatus, TaskFailureKind, TaskResult
from ..cache_task_id import conversation_client
from ..node_proxy_preload import install_proxy_preload, with_proxy_preload
from ..trace_spans import brain_turn_attributes


logger = logging.getLogger(__name__)
# NOTE(wallyshi): The real CLI takes ~10s to confirm an ACP cancel; the runtime quarantines a session
# whose cancel is unconfirmed after the grace period, so the default of a few seconds is too short.
_INTERRUPT_GRACE_TIMEOUT = 20.0

_TRAILING_FRAME_TIMEOUT_S = 2.0
"""How long a returned turn waits for pushed frames its result counted but the Host has not received."""

PROMPT_BUDGET_S = 2000.0
"""Agent working time of one turn; time the turn spends waiting on the user does not count."""
_RUNTIME_FREEZE_ALLOWANCE_S = 4 * 3600.0
"""Frozen time the sandbox's own prompt timer leaves room for on top of the wall-clock cap.

The sandbox cannot see the episode freeze, so its timer only backs up a Host that went away; hylive's
wall-clock timer, which skips frozen stretches, ends the turn and interrupts it.
"""
_TURN_SETTLE_TIMEOUT_S = 30.0
"""How long an interrupt may take to be confirmed."""
_CLOSE_TURN_WAIT_S = 5.0
"""How long closing a session waits for a turn the host cancelled to come back."""
_DEFAULT_PERMISSION_POLICY = CodeBuddyMultiturnInput.model_fields['permission_policy'].default
_DEFAULT_LLM_CLIENT_NAME = CodeBuddyMultiturnInput.model_fields['llm_client_name'].default
_TASK_OWNED_INPUT_FIELDS = frozenset({'user_prompt', 'timeout', 'max_turns', 'get_user_input'})

AgentOutputT = TypeVar('AgentOutputT', bound=AgentRunOutput)
TurnEndedSink = Callable[[str, str, TaskResult], Awaitable[None]]
"""Receives ``(turn_id, session_id, result)`` once a turn returned."""


class WorkAgentBudget(BaseModel):
    """Bound one working-agent execution.

    Unknown keys are ignored.
    """

    timeout_s: float = Field(default=3600.0, gt=0)
    """Wall-clock cap of one execution, time waiting on the user included.

    Time the episode is frozen on a paused cerebellum does not count. A turn is governed by its shorter
    working budget, which also excludes waiting on the user; this cap is the safety net behind it.
    """


class AcpRequestBinder(Protocol):
    """Where a turn's ACP permission and elicitation requests go: bound by channel, for the turn."""

    def bind(self, channel_id: str, *, owner_id: str, turn_id: str) -> None: ...

    def unbind(self, channel_id: str) -> None: ...

    def listen(self, runtime: Any) -> None: ...


async def _run_with_acp_updates(
    run: Awaitable[TaskResult],
    *,
    update_router: AcpUpdateRouter,
    runtime: Any,
    channel_id: str,
    relay_turn_id: str,
    owner_id: str,
    turn_id: str,
) -> TaskResult:
    """Await one turn, routing its raw ACP updates live and, if the push delivered none, from its events.

    While the turn runs, frames the runtime pushes on ``channel_id`` stamped with ``relay_turn_id`` reach
    hylive's turn ``turn_id``. Once it returns the channel is unbound, so a CLI that outlived the turn
    cannot write to it. The turn's own events are replayed only when no frame arrived; a push that fell
    short of ``relay_frames`` is only logged.
    """
    update_router.listen(runtime)
    update_router.bind(channel_id, turn_id=relay_turn_id, owner_id=owner_id, task_id=turn_id)
    try:
        result = await run
    except BaseException:
        update_router.unbind(channel_id)
        raise
    events = result.metadata.pop('acp_events', None)
    relay_frames = result.metadata.pop('acp_relay_frames', None)
    try:
        if relay_frames is not None:
            # NOTE(wallyshi): Frames and the invoke result reach the Host on different paths, so the last
            # frames can trail the result; replaying instead would reorder a concurrent stream's records.
            await update_router.wait_for_frames(channel_id, relay_frames, timeout=_TRAILING_FRAME_TIMEOUT_S)
    finally:
        received = update_router.unbind(channel_id) or 0
    events = events if isinstance(events, list) else []
    if events and received == 0:
        update_router.replay(owner_id, turn_id, events)
    elif events and relay_frames is not None and received < relay_frames:
        # NOTE(wallyshi): seq counts only updates that survive trim, so it cannot locate the lost frames
        # in the untrimmed events; a partial replay would write the delivered ones twice.
        logger.warning(
            'Raw ACP updates lost channel=%s turn=%s pushed=%s relayed=%s', channel_id, turn_id, received, relay_frames
        )
    return result


class AcpSessionExecutor:
    """Carry out hylive's ACP primitives on the sandbox's CodeBuddy or WorkBuddy sessions.

    ``trusted_config['acp_cli']`` selects the CLI; ``workbuddy`` steers and answers btw questions.
    """

    def __init__(
        self,
        *,
        ctx: ExecutionContext,
        update_router: AcpUpdateRouter,
        turn_ended_sink: TurnEndedSink,
        request_binder: AcpRequestBinder | None = None,
        trusted_config: dict[str, Any] | None = None,
        budget: WorkAgentBudget | None = None,
        session_cache_task_id: bool = False,
        owner_of: Callable[[str], str] | None = None,
        turn_settle_timeout_s: float = _TURN_SETTLE_TIMEOUT_S,
    ) -> None:
        self.ctx = ctx
        self.update_router = update_router
        self.turn_ended_sink = turn_ended_sink
        self.request_binder = request_binder
        self.trusted_config = dict(trusted_config or {})
        self.acp_cli: AcpCli = self.trusted_config.get('acp_cli', 'codebuddy')
        self.budget = budget or WorkAgentBudget()
        self.session_cache_task_id = session_cache_task_id
        """Give every project Session a model client with its own ``cache_task_id``."""
        self.owner_of = owner_of
        """The operation a turn works for, named on the turn's trace spans."""
        self.turn_settle_timeout_s = turn_settle_timeout_s
        self._proxy_preload: asyncio.Task[None] | None = None
        self._turns: dict[str, asyncio.Task[None]] = {}
        """The turn each session runs, by session id."""
        self._sessions: dict[str, SessionRef] = {}
        """Sessions a turn ran on and nobody closed yet; ``aclose`` closes what hylive did not."""

    def supports_steer(self) -> bool:
        """WorkBuddy can insert a new request into the running turn; CodeBuddy cancels and resends."""
        return self.acp_cli == 'workbuddy'

    def supports_side_question(self) -> bool:
        """WorkBuddy answers a question with btw, beside the session and without tools."""
        return self.acp_cli == 'workbuddy'

    def registers_artifacts_from_acp(self) -> bool:
        """WorkBuddy names its deliverables in ACP updates; CodeBuddy's files are found by listing."""
        return self.acp_cli == 'workbuddy'

    async def warm_up(self) -> None:
        """Start the sandbox and install the proxy preload, so the first turn does not wait on them."""
        await self._ensure_proxy_preload()

    async def open_session(self, session: SessionRef) -> None:
        """A session's CLI starts with its first turn; opening only makes sure the sandbox is ready for it."""
        await self._ensure_proxy_preload()

    async def prompt(self, session: SessionRef, *, turn_id: str, prompt: str) -> None:
        """Start one turn in the background and return; its end goes to ``turn_ended_sink``."""
        self._sessions[session.session_id] = session
        task = asyncio.create_task(
            self._run_turn(session, turn_id=turn_id, prompt=prompt), name=f'hy-live-turn:{session.session_id}'
        )
        self._turns[session.session_id] = task
        task.add_done_callback(
            lambda finished, session_id=session.session_id: self._turn_finished(session_id, finished)
        )

    async def interrupt(self, session: SessionRef) -> None:
        """Interrupt the session's turn.

        Raises:
            RuntimeError: The CLI did not confirm it stopped within ``turn_settle_timeout_s``.
        """
        try:
            await asyncio.wait_for(self._cancel(session), timeout=self.turn_settle_timeout_s)
        except TimeoutError as error:
            raise RuntimeError(f'interrupt of {session.session_id} not confirmed in time') from error

    async def steer(self, session: SessionRef, *, prompt: str) -> dict[str, Any]:
        """Ask the runtime once to insert ``prompt`` into the running turn; hylive retries.

        Returns:
            ``{steered, error_code, stderr}``; a runtime image without steering is a steer that did not land.
        """
        runtime = await self.ctx.get_runtime()
        try:
            result = await runtime.invoke(
                InvokeRequest.build(
                    steer_codebuddy_session,
                    venv_id=runtime.default_venv_id,
                    session_id=_channel_id(session),
                    prompt=prompt,
                )
            )
        except Exception as error:  # noqa: BLE001
            logger.warning('ACP steer is unavailable session=%s', session.session_id, exc_info=True)
            return {'steered': False, 'error_code': None, 'stderr': str(error) or type(error).__name__}
        detail: dict[str, Any] = result if isinstance(result, dict) else {}
        return {
            'steered': bool(detail.get('steered')),
            'error_code': detail.get('error_code'),
            'stderr': str(detail.get('stderr') or ''),
        }

    async def side_question(self, session: SessionRef, *, question: str) -> dict[str, Any]:
        """Ask one btw question beside the session; the runtime's ``{success, answer, stderr, error_code}``."""
        try:
            runtime = await self.ctx.get_runtime()
            result = await runtime.invoke(
                InvokeRequest.build(
                    side_question_codebuddy_session,
                    venv_id=runtime.default_venv_id,
                    session_id=_channel_id(session),
                    question=question,
                )
            )
        except Exception as error:  # noqa: BLE001
            logger.exception('Side question failed session=%s', session.session_id)
            return {'success': False, 'answer': '', 'stderr': str(error) or type(error).__name__}
        return result if isinstance(result, dict) else {'success': False, 'answer': '', 'stderr': repr(result)}

    async def close_session(self, session: SessionRef) -> None:
        """Close the sandbox session; a turn still running on the host is cancelled first.

        Raises:
            RuntimeError: The runtime did not close the session.
        """
        self._sessions.pop(session.session_id, None)
        turn = self._turns.pop(session.session_id, None)
        if turn is not None and not turn.done():
            turn.cancel()
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(turn), timeout=_CLOSE_TURN_WAIT_S)
        channel_id = _channel_id(session)
        try:
            runtime = await self.ctx.get_runtime()
        except Exception:
            if self.request_binder is not None:
                self.request_binder.unbind(channel_id)
            raise
        try:
            result = await call_runtime_agent(close_codebuddy_session, session_id=channel_id)
        finally:
            unbind_llm_channel(runtime, channel_id)
            if self.request_binder is not None:
                self.request_binder.unbind(channel_id)
        if not result.get('success'):
            raise RuntimeError(str(result.get('stderr') or f'Failed to close {session.session_id}'))

    async def aclose(self) -> None:
        """Cancel the turns still running on the host and close the sessions hylive did not; failures are logged."""
        turns = [turn for turn in self._turns.values() if not turn.done()]
        for turn in turns:
            turn.cancel()
        await asyncio.gather(*turns, return_exceptions=True)
        self._turns.clear()
        for session in list(self._sessions.values()):
            try:
                await self.close_session(session)
            except Exception:  # noqa: BLE001
                logger.exception('Failed to close the project Session %s', session.session_id)
        if self._proxy_preload is not None and not self._proxy_preload.done():
            self._proxy_preload.cancel()
            await asyncio.gather(self._proxy_preload, return_exceptions=True)

    def _turn_finished(self, session_id: str, finished: asyncio.Task[None]) -> None:
        if self._turns.get(session_id) is finished:
            del self._turns[session_id]
        if not finished.cancelled() and (error := finished.exception()) is not None:
            logger.error('Project turn failed session=%s', session_id, exc_info=error)

    async def _run_turn(self, session: SessionRef, *, turn_id: str, prompt: str) -> None:
        relay_turn_id = uuid.uuid4().hex
        config = {
            **self.trusted_config,
            'session_id': session.session_id,
            'working_dir': session.working_dir,
            'close_on_finish': False,
            'relay_turn_id': relay_turn_id,
        }
        if self.session_cache_task_id:
            config['llm_client_name'] = self._session_llm_client(
                session, config.get('llm_client_name', _DEFAULT_LLM_CLIENT_NAME)
            )
        if self.request_binder is not None and config.get('permission_policy', _DEFAULT_PERMISSION_POLICY) == 'defer':
            # NOTE(wallyshi): the permission relay also answers CodeBuddy's questions, so it may always ask.
            config.setdefault('client_questions', True)
        channel_id = _channel_id(session)
        try:
            operation_id = self.owner_of(turn_id) if self.owner_of is not None else turn_id
            runtime = await self.ctx.get_runtime()
            if self.request_binder is not None:
                self.request_binder.bind(channel_id, owner_id=session.owner_id, turn_id=turn_id)
                self.request_binder.listen(runtime)
            # The turn's task copies this context, so every span of the turn, its model calls included,
            # names the operation it works for.
            with trace_attributes_context(
                brain_turn_attributes(operation_id=operation_id, session_id=session.session_id, prompt=prompt)
            ):
                turn = asyncio.create_task(
                    _run_with_acp_updates(
                        self._run_codebuddy(prompt, config),
                        update_router=self.update_router,
                        runtime=runtime,
                        channel_id=channel_id,
                        relay_turn_id=relay_turn_id,
                        owner_id=session.owner_id,
                        turn_id=turn_id,
                    )
                )
            try:
                result = await asyncio.shield(turn)
            except asyncio.CancelledError:
                turn.cancel()
                await asyncio.gather(turn, return_exceptions=True)
                raise
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001
            task_error = TaskError(code='PROJECT_EXCEPTION', message=str(error) or type(error).__name__)
            result = TaskResult(
                execution_status='failed', failure_kind='infrastructure', error=task_error, partial=True
            )
        try:
            await self.turn_ended_sink(turn_id, session.session_id, result)
        except Exception:  # noqa: BLE001
            logger.exception('Failed to publish the end of turn %s session=%s', turn_id, session.session_id)

    async def _run_codebuddy(self, prompt: str, config: dict[str, Any]) -> TaskResult:
        """Run one turn; a failure is returned as a result, never raised. hylive's timers cap it."""
        try:
            return await self._invoke_codebuddy(prompt, config)
        except Exception as error:  # noqa: BLE001
            return _execution_failure(error)

    async def _invoke_codebuddy(self, prompt: str, config: dict[str, Any]) -> TaskResult:
        """Send ``prompt`` as one ACP turn of the configured CodeBuddy or WorkBuddy session."""
        await self._ensure_proxy_preload()
        config = {**config, 'extra_env': with_proxy_preload(config.get('extra_env'))}
        if self.acp_cli == 'workbuddy':
            config['extra_env']['NEXUS_WORKBUDDY_PROXY_CHILDREN'] = '1'
        agent_input = _codebuddy_input(prompt, config, timeout_s=self.budget.timeout_s + _RUNTIME_FREEZE_ALLOWANCE_S)
        output, trajectory_count, trajectory = await _run_with_chat_tracker(run_codebuddy_multiturn(agent_input))
        return _codebuddy_result(output, trajectory_count, trajectory)

    async def _cancel(self, session: SessionRef) -> None:
        runtime = await self.ctx.get_runtime()
        result = await runtime.invoke(
            InvokeRequest.build(
                interrupt_codebuddy_session,
                venv_id=runtime.default_venv_id,
                session_id=_channel_id(session),
                grace_timeout=_INTERRUPT_GRACE_TIMEOUT,
            )
        )
        if not result.get('success'):
            raise RuntimeError(str(result.get('stderr') or f'Failed to cancel {session.session_id}'))

    async def _ensure_proxy_preload(self) -> None:
        """Install the Node proxy preload once per sandbox; concurrent turns share the install.

        A failed or cancelled install is retried by the next turn.
        """
        install = self._proxy_preload
        if install is None or (install.done() and (install.cancelled() or install.exception() is not None)):
            install = self._proxy_preload = asyncio.create_task(self._install_proxy_preload())
        await asyncio.shield(install)

    async def _install_proxy_preload(self) -> None:
        await install_proxy_preload(await self.ctx.get_runtime())

    def _session_llm_client(self, session: SessionRef, base_name: str) -> str:
        """Register, once per Session, a copy of ``base_name`` with its own ``cache_task_id``; return its name.

        Every turn of the Session gets the same instance, as its persistent LLM channel requires.
        """
        name = f'{base_name}@{_channel_id(session)}'
        if name not in self.ctx.openai_clients:
            self.ctx.openai_clients[name] = conversation_client(
                self.ctx.get_openai_client(base_name),
                conversation='backend',
                enabled=True,
            )
        return name


def _channel_id(session: SessionRef) -> str:
    return normalize_stream_json_session_id(session.session_id, namespace='codebuddy-acp-session')


async def _end_after_single_turn(_request: Any) -> None:
    return None


def _codebuddy_input(prompt: str, config: dict[str, Any], *, timeout_s: float) -> CodeBuddyMultiturnInput:
    conflicting_keys = sorted(_TASK_OWNED_INPUT_FIELDS.intersection(config))
    if conflicting_keys:
        raise ValueError(f'Working-agent config must not override task-owned fields: {conflicting_keys}')
    config = dict(config)
    config.setdefault('llm_client_name', 'external')
    if config.get('skip_permissions') is False:
        config.setdefault('permission_policy', 'defer')
    if config.get('permission_policy') == 'defer':
        # NOTE(wallyshi): deferred approvals wait on the user, so no runtime timer may cancel them.
        config.setdefault('deferred_permission_timeout', None)
    return CodeBuddyMultiturnInput.model_validate({
        **config,
        'user_prompt': prompt,
        'timeout': timeout_s,
        'max_turns': 1,
        'get_user_input': _end_after_single_turn,
    })


def _codebuddy_result(output: Any, trajectory_count: int, trajectory: list[dict[str, Any]]) -> TaskResult:
    """Map one CLI turn to a task result."""
    metadata = {
        'exit_code': output.exit_code,
        'trajectory_count': trajectory_count,
        'session_id': output.session_id,
        'session_status': output.status,
        'turns': output.turns,
        # _run_with_acp_updates pops these and replays them to the update router when the push delivered
        # none, so they never stay in the task result.
        'acp_events': list(output.events),
        'acp_relay_frames': output.relay_frames,
    }
    if output.status == 'cancelled':
        return _unsuccessful_result(
            'cancelled',
            'cancel',
            code='WORK_AGENT_CANCELLED',
            stage=output.failure_stage or 'execute',
            message=output.stderr or 'CodeBuddy execution was cancelled',
            final_output=output.output,
            trajectory=trajectory,
            metadata=metadata,
        )
    if output.status == 'timeout':
        return _unsuccessful_result(
            'timeout',
            'timeout',
            code='WORK_AGENT_TIMEOUT',
            stage=output.failure_stage or 'execute',
            message=output.stderr or 'CodeBuddy execution timed out',
            retryable=True,
            final_output=output.output,
            trajectory=trajectory,
            metadata=metadata,
        )
    if output.success and trajectory:
        return TaskResult(final_output=output.output, trajectory=list(trajectory), metadata=metadata)
    if output.success:
        code, stage, message = 'EMPTY_TRAJECTORY', 'collect', 'CodeBuddy returned no model-facing trajectory'
    else:
        code = 'CODEBUDDY_FAILED'
        stage = output.failure_stage or 'execute'
        message = output.stderr or f'CodeBuddy exited with code {output.exit_code}'
    # NOTE(wallyshi): A turn CodeBuddy finished cleanly is a success even when it did not get the work done;
    # the reward judges that. Getting here means the CLI or its model call broke (non-zero exit, an
    # is_error result such as an internal 429, no trajectory), which the trainer must mask, not score.
    return _unsuccessful_result(
        'failed',
        'infrastructure',
        code=code,
        stage=stage,
        message=message,
        final_output=output.output,
        trajectory=trajectory,
        metadata=metadata,
    )


def _execution_failure(error: BaseException) -> TaskResult:
    logger.exception('CodeBuddy turn failed during execution')
    return _unsuccessful_result(
        'failed',
        'infrastructure',
        code='WORK_AGENT_EXECUTION_FAILED',
        stage='execute',
        message=str(error) or type(error).__name__,
    )


def _unsuccessful_result(
    execution_status: TaskExecutionStatus,
    failure_kind: TaskFailureKind,
    *,
    code: str,
    stage: str,
    message: str,
    retryable: bool = False,
    final_output: str = '',
    trajectory: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> TaskResult:
    return TaskResult(
        execution_status=execution_status,
        failure_kind=failure_kind,
        error=TaskError(
            code=code,
            message=message,
            retryable=retryable,
            details={'stage': stage, 'failure_kind': failure_kind},
        ),
        final_output=final_output,
        trajectory=list(trajectory or []),
        metadata=dict(metadata or {}),
        partial=execution_status in {'timeout', 'cancelled'},
    )


async def _run_with_chat_tracker(
    invocation: Awaitable[AgentOutputT],
) -> tuple[AgentOutputT, int, list[dict[str, Any]]]:
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    with processor_context([tracker], finalize_on_exit=True):
        output = await invocation
    raw_trajectories = tracker.get_result().get('trajectories') or []
    trajectories = [trajectory for trajectory in raw_trajectories if isinstance(trajectory, dict)]
    selected = _select_trajectory(trajectories)
    return output, len(trajectories), selected.get('messages', []) if selected else []


def _select_trajectory(trajectories: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the longest completed branch, then the longest partial branch."""
    candidates = [trajectory for trajectory in trajectories if _trajectory_messages(trajectory)]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda trajectory: (
            _is_complete_trajectory(trajectory),
            _count_assistant_turns(_trajectory_messages(trajectory)),
            len(_trajectory_messages(trajectory)),
        ),
    )


def _trajectory_messages(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    messages = trajectory.get('messages')
    if not isinstance(messages, list):
        return []
    return [message for message in messages if isinstance(message, dict)]


def _is_complete_trajectory(trajectory: dict[str, Any]) -> bool:
    messages = _trajectory_messages(trajectory)
    if not messages:
        return False
    last_message = messages[-1]
    return last_message.get('role') == 'assistant' and not last_message.get('tool_calls')


def _count_assistant_turns(trajectory: list[dict[str, Any]]) -> int:
    return sum(message.get('role') == 'assistant' for message in trajectory)


__all__ = ['PROMPT_BUDGET_S', 'AcpSessionExecutor', 'WorkAgentBudget']
