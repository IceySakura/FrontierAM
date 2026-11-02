"""ACP primitives on CodeBuddy and WorkBuddy sessions: turns, their updates and results, steers, btw questions."""

from __future__ import annotations

import asyncio
import logging

from types import SimpleNamespace
from typing import Any, cast

import pytest

from nexus.extensions.agents.installed_common.multiturn import normalize_stream_json_session_id
from nexus.extensions.projects.hy_live import agent as agent_module
from nexus.extensions.projects.hy_live.coordination import backend as backend_module
from nexus.extensions.projects.hy_live.coordination.acp_updates import AcpUpdateRouter
from nexus.extensions.projects.hy_live.coordination.backend import (
    AcpSessionExecutor,
    WorkAgentBudget,
    _run_with_acp_updates,
)
from nexus.extensions.projects.hy_live.coordination.models import SessionRef, TaskResult


ReceivedUpdate = tuple[str, str, dict[str, Any]]
Ended = tuple[str, str, TaskResult]
_REAL_PROXY_PRELOAD_INSTALL = AcpSessionExecutor._install_proxy_preload
_WORKBUDDY = {'acp_cli': 'workbuddy'}
_SESSION = SessionRef(session_id='project-1', owner_id='owner-1', working_dir='/workspace')
_CHANNEL = normalize_stream_json_session_id('project-1', namespace='codebuddy-acp-session')
_PRELOAD_OPTION = '--require /tmp/nexus-node-proxy/preload.js'


@pytest.fixture(autouse=True)
def _skip_proxy_preload_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most executors here have no sandbox; the preload tests put the real install back."""

    async def installed(_executor: AcpSessionExecutor) -> None:
        return None

    monkeypatch.setattr(AcpSessionExecutor, '_install_proxy_preload', installed)


@pytest.fixture
def real_preload_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AcpSessionExecutor, '_install_proxy_preload', _REAL_PROXY_PRELOAD_INSTALL)


def _tool_update(call_id: str, status: str, command: str = 'ls') -> dict[str, Any]:
    kind = 'tool_call' if status == 'pending' else 'tool_call_update'
    return {
        'sessionUpdate': kind,
        'toolCallId': call_id,
        'status': status,
        'title': f'`{command}`',
        'rawInput': {'command': command},
    }


def _message_update(text: str) -> dict[str, Any]:
    return {'sessionUpdate': 'agent_message_chunk', 'content': {'type': 'text', 'text': text}, 'messageId': 'm1'}


def _frame(channel_id: str, seq: int, update: dict[str, Any], turn_id: str = 'relay-1') -> dict[str, Any]:
    return {'channel_id': channel_id, 'turn_id': turn_id, 'seq': seq, 'update': update}


def _tool_statuses(received: list[ReceivedUpdate]) -> list[tuple[str, str | None, str]]:
    return [
        (turn_id, update.get('toolCallId'), update['status'])
        for _, turn_id, update in received
        if update.get('sessionUpdate') in ('tool_call', 'tool_call_update')
    ]


def _recording_router(received: list[ReceivedUpdate]) -> AcpUpdateRouter:
    return AcpUpdateRouter(lambda owner, turn, update: received.append((owner, turn, update)))


def _result(events: list[dict[str, Any]], status: str = 'success', *, relay_frames: int | None = None) -> TaskResult:
    metadata = {
        'session_id': 's',
        'acp_events': [{'sessionId': 'remote', 'update': event} for event in events],
        'acp_relay_frames': relay_frames,
    }
    if status == 'success':
        return TaskResult(final_output='done', metadata=metadata)
    return backend_module._unsuccessful_result(
        status, 'timeout', code='TURN_TIMEOUT', stage='execute', message='turn timeout', metadata=metadata
    )


class _SilentRuntime:
    def on(self, event: str, listener: Any) -> None:
        return None


async def _run_turn(run: Any, router: AcpUpdateRouter) -> TaskResult:
    return await _run_with_acp_updates(
        run,
        update_router=router,
        runtime=_SilentRuntime(),
        channel_id='chan-a',
        relay_turn_id='relay-1',
        owner_id='owner-1',
        turn_id='turn-1',
    )


class TestRunWithAcpUpdates:
    async def test_pushed_frames_arrive_while_the_turn_runs(self) -> None:
        received: list[ReceivedUpdate] = []
        router = _recording_router(received)
        events = [_tool_update('c1', 'pending'), _tool_update('c1', 'completed'), _message_update('Done.')]
        seen_while_running: list[int] = []

        async def run() -> TaskResult:
            for seq, event in enumerate(events, start=1):
                router.dispatch(_frame('chan-a', seq, event))
            seen_while_running.append(len(received))
            return _result(events, relay_frames=3)

        result = await _run_turn(run(), router)

        assert seen_while_running == [3], 'every pushed frame reaches the turn before it returns'
        assert [update for _, _, update in received] == events, 'a fully pushed turn needs no replay'
        assert 'acp_events' not in result.metadata, 'the raw ACP stream must not stay in the task result'
        assert 'acp_relay_frames' not in result.metadata

    async def test_a_partially_pushed_turn_is_logged_not_replayed(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(backend_module, '_TRAILING_FRAME_TIMEOUT_S', 0.05)
        received: list[ReceivedUpdate] = []
        router = _recording_router(received)
        events = [_tool_update('c1', 'pending'), _tool_update('c1', 'completed')]

        async def run() -> TaskResult:
            router.dispatch(_frame('chan-a', 1, events[0]))
            return _result(events, relay_frames=2)

        with caplog.at_level(logging.WARNING, logger=backend_module.__name__):
            await _run_turn(run(), router)

        assert _tool_statuses(received) == [('turn-1', 'c1', 'pending')]
        assert 'Raw ACP updates lost' in caplog.text

    async def test_a_frame_that_trails_the_turn_result_is_waited_for_instead_of_replayed(self) -> None:
        received: list[ReceivedUpdate] = []
        router = _recording_router(received)
        events = [_tool_update('c1', 'pending'), _tool_update('c1', 'completed')]

        async def run() -> TaskResult:
            router.dispatch(_frame('chan-a', 1, events[0]))
            asyncio.get_running_loop().call_later(0.01, router.dispatch, _frame('chan-a', 2, events[1]))
            return _result(events, relay_frames=2)

        await _run_turn(run(), router)

        assert _tool_statuses(received) == [('turn-1', 'c1', 'pending'), ('turn-1', 'c1', 'completed')], 'no replay'

    async def test_a_raising_turn_unbinds_its_channel(self) -> None:
        received: list[ReceivedUpdate] = []
        router = _recording_router(received)

        async def run() -> TaskResult:
            router.dispatch(_frame('chan-a', 1, _tool_update('c1', 'pending', command='sleep 120')))
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await _run_turn(run(), router)

        assert _tool_statuses(received) == [('turn-1', 'c1', 'pending')]
        assert router.unbind('chan-a') is None

    async def test_frames_after_the_turn_returned_are_dropped(self) -> None:
        received: list[ReceivedUpdate] = []
        router = _recording_router(received)

        async def run() -> TaskResult:
            return _result([])

        await _run_turn(run(), router)
        router.dispatch(_frame('chan-a', 1, _tool_update('c9', 'pending')))

        assert received == []

    async def test_without_a_push_the_turn_events_are_replayed_in_order(self) -> None:
        received: list[ReceivedUpdate] = []
        events = [_tool_update('c1', 'pending'), _tool_update('c1', 'failed')]

        async def run() -> TaskResult:
            return _result(events, status='timeout')

        await _run_turn(run(), _recording_router(received))

        assert received == [('owner-1', 'turn-1', event) for event in events]


class SandboxRuntime:
    """A sandbox runtime: invokes answer from ``results`` (a dict, an exception, or a coroutine factory)."""

    default_venv_id = 'venv'

    def __init__(self, results: list[Any] | None = None) -> None:
        self.results = list(results or [])
        self.calls: list[dict[str, Any]] = []
        self.handlers: dict[str, Any] = {}

    def on(self, event: str, handler: Any) -> None:
        self.handlers[event] = handler

    async def invoke(self, request: Any) -> Any:
        _args, kwargs = request.unpack_payload()
        self.calls.append(kwargs)
        result = self.results.pop(0) if self.results else {'success': True}
        if isinstance(result, BaseException):
            raise result
        if callable(result):
            return await result()
        return result


class RecordingBinder:
    def __init__(self) -> None:
        self.binds: list[tuple[str, str]] = []
        self.unbinds: list[str] = []

    def bind(self, channel_id: str, *, owner_id: str, turn_id: str) -> None:
        self.binds.append((channel_id, turn_id))

    def unbind(self, channel_id: str) -> None:
        self.unbinds.append(channel_id)

    def listen(self, runtime: Any) -> None:
        return None


def _executor(
    *,
    runtime: Any = None,
    trusted_config: dict[str, Any] | None = None,
    ended: list[Ended] | None = None,
    received: list[ReceivedUpdate] | None = None,
    binder: Any = None,
    **kwargs: Any,
) -> AcpSessionExecutor:
    runtime = runtime if runtime is not None else SandboxRuntime()

    async def get_runtime() -> Any:
        return runtime

    async def turn_ended(turn_id: str, session_id: str, result: TaskResult) -> None:
        if ended is not None:
            ended.append((turn_id, session_id, result))

    return AcpSessionExecutor(
        ctx=cast(Any, SimpleNamespace(run_id='run-1', get_runtime=get_runtime, openai_clients={})),
        update_router=_recording_router(received if received is not None else []),
        turn_ended_sink=turn_ended,
        request_binder=binder,
        trusted_config=trusted_config,
        **kwargs,
    )


async def _until(predicate: Any) -> None:
    for _ in range(500):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError('condition never held')


def _codebuddy_output(**overrides: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        'success': True,
        'output': 'done',
        'stderr': '',
        'exit_code': 0,
        'execution_time': 1.0,
        'failure_kind': None,
        'failure_stage': None,
        'session_id': 'normalized-session',
        'status': 'ok',
        'turns': 1,
        'events': [],
        'relay_frames': None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _patch_codebuddy(
    monkeypatch: pytest.MonkeyPatch, output: SimpleNamespace, trajectory: list[dict[str, Any]]
) -> list[Any]:
    inputs: list[Any] = []

    async def run_codebuddy_multiturn(agent_input: Any) -> SimpleNamespace:
        inputs.append(agent_input)
        return output

    async def run_with_tracker(invocation: Any) -> tuple[Any, int, list[dict[str, Any]]]:
        return await invocation, len(trajectory), trajectory

    monkeypatch.setattr(backend_module, 'run_codebuddy_multiturn', run_codebuddy_multiturn)
    monkeypatch.setattr(backend_module, '_run_with_chat_tracker', run_with_tracker)
    return inputs


async def _one_turn(executor: AcpSessionExecutor, ended: list[Ended], *, session: SessionRef = _SESSION) -> TaskResult:
    await executor.prompt(session, turn_id='turn-1', prompt='build project')
    await _until(lambda: ended)
    ((turn_id, session_id, result),) = ended
    assert (turn_id, session_id) == ('turn-1', session.session_id)
    return result


async def test_a_prompt_runs_in_the_background_relays_its_updates_and_reports_its_end(
    fake_runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: list[ReceivedUpdate] = []
    ended: list[Ended] = []
    first_frame_seen = asyncio.Event()
    release = asyncio.Event()

    async def push_frames(prompt: str, config: dict[str, Any]) -> TaskResult:
        listener = fake_runtime.handlers['acp_session_update']
        relay = config['relay_turn_id']
        listener(_frame(_CHANNEL, 1, _tool_update('c1', 'pending', command='echo first'), turn_id=relay))
        first_frame_seen.set()
        await release.wait()
        listener(_frame(_CHANNEL, 2, _tool_update('c1', 'completed', command='echo first'), turn_id=relay))
        return _result(
            [_tool_update('c1', 'pending', 'echo first'), _tool_update('c1', 'completed', 'echo first')],
            relay_frames=2,
        )

    executor = _executor(runtime=fake_runtime, ended=ended, received=received)
    monkeypatch.setattr(executor, '_invoke_codebuddy', push_frames)

    await executor.prompt(_SESSION, turn_id='turn-7', prompt='long work')
    await asyncio.wait_for(first_frame_seen.wait(), timeout=1)

    assert _tool_statuses(received) == [('turn-7', 'c1', 'pending')], 'updates name hylive turn, not the relay id'
    assert ended == [], 'the first step is visible while the turn still runs'
    release.set()
    await _until(lambda: ended)
    assert _tool_statuses(received) == [('turn-7', 'c1', 'pending'), ('turn-7', 'c1', 'completed')]
    ((turn_id, _, result),) = ended
    assert turn_id == 'turn-7'
    assert 'acp_events' not in result.metadata
    assert 'work_agent_status' not in result.metadata, 'hylive stamps the turn result itself'


async def test_a_turn_sends_one_codebuddy_prompt_on_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    trajectory = [{'role': 'assistant', 'content': 'done'}]
    inputs = _patch_codebuddy(monkeypatch, _codebuddy_output(), trajectory)
    ended: list[Ended] = []
    binder = RecordingBinder()
    executor = _executor(ended=ended, budget=WorkAgentBudget(timeout_s=30), binder=binder)

    result = await _one_turn(executor, ended, session=SessionRef(session_id='project-1', owner_id='o', working_dir='/w/r'))

    (agent_input,) = inputs
    assert agent_input.working_dir == '/w/r'
    assert agent_input.user_prompt == 'build project'
    assert agent_input.timeout == 30 + backend_module._RUNTIME_FREEZE_ALLOWANCE_S
    assert agent_input.max_turns == 1
    assert agent_input.close_on_finish is False
    assert agent_input.session_id == 'project-1'
    assert agent_input.llm_client_name == 'external'
    assert agent_input.relay_turn_id
    assert binder.binds == [(_CHANNEL, 'turn-1')], 'ACP requests of the turn are bound to hylive turn'
    assert result.execution_status == 'success'
    assert result.trajectory == trajectory
    assert result.metadata['session_id'] == 'normalized-session'


@pytest.mark.parametrize(
    ('config', 'expected_timeout'),
    [({'skip_permissions': False}, None), ({'skip_permissions': False, 'deferred_permission_timeout': 60}, 60)],
)
async def test_a_deferring_turn_waits_for_the_gateway(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, Any], expected_timeout: float | None
) -> None:
    inputs = _patch_codebuddy(monkeypatch, _codebuddy_output(), [{'role': 'assistant', 'content': 'done'}])
    ended: list[Ended] = []

    await _one_turn(_executor(ended=ended, trusted_config=config, binder=RecordingBinder()), ended)

    assert inputs[0].permission_policy == 'defer'
    assert inputs[0].deferred_permission_timeout == expected_timeout
    assert inputs[0].client_questions is True, 'the relay also answers CodeBuddy questions'


@pytest.mark.parametrize(
    ('output', 'trajectory', 'expected'),
    [
        (
            _codebuddy_output(status='timeout', stderr='prompt timed out'),
            [{'role': 'assistant', 'content': 'partial'}],
            ('timeout', 'timeout', 'WORK_AGENT_TIMEOUT', 'execute', True),
        ),
        (
            _codebuddy_output(status='cancelled', output='partial work before cancel'),
            [{'role': 'assistant', 'content': 'partial work before cancel'}],
            ('cancelled', 'cancel', 'WORK_AGENT_CANCELLED', 'execute', True),
        ),
        (
            _codebuddy_output(success=False, status='failed', exit_code=2, failure_stage='prompt'),
            [],
            ('failed', 'infrastructure', 'CODEBUDDY_FAILED', 'prompt', False),
        ),
        (_codebuddy_output(), [], ('failed', 'infrastructure', 'EMPTY_TRAJECTORY', 'collect', False)),
    ],
)
async def test_a_turn_maps_its_outcome_to_the_task_result(
    monkeypatch: pytest.MonkeyPatch,
    output: SimpleNamespace,
    trajectory: list[dict[str, Any]],
    expected: tuple[str, str, str, str, bool],
) -> None:
    execution_status, failure_kind, error_code, stage, partial = expected
    _patch_codebuddy(monkeypatch, output, trajectory)
    ended: list[Ended] = []

    result = await _one_turn(_executor(ended=ended), ended)

    assert result.execution_status == execution_status
    assert result.failure_kind == failure_kind
    assert result.error is not None
    assert result.error.code == error_code
    assert result.error.details == {'stage': stage, 'failure_kind': failure_kind}
    assert result.final_output == output.output
    assert result.trajectory == trajectory
    assert result.partial is partial


async def test_a_raising_turn_is_an_infrastructure_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    ended: list[Ended] = []
    executor = _executor(ended=ended)

    async def disconnect(prompt: str, config: dict[str, Any]) -> TaskResult:
        raise RuntimeError('codebuddy disconnected')

    monkeypatch.setattr(executor, '_invoke_codebuddy', disconnect)

    result = await _one_turn(executor, ended)

    assert result.execution_status == 'failed'
    assert result.failure_kind == 'infrastructure'
    assert result.error is not None
    assert result.error.code == 'WORK_AGENT_EXECUTION_FAILED'
    assert result.error.message == 'codebuddy disconnected'


async def test_a_config_overriding_a_task_owned_input_field_fails_the_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _patch_codebuddy(monkeypatch, _codebuddy_output(), [{'role': 'assistant', 'content': 'done'}])
    ended: list[Ended] = []

    result = await _one_turn(_executor(ended=ended, trusted_config={'timeout': 5}), ended)

    assert inputs == []
    assert result.execution_status == 'failed'
    assert result.error is not None
    assert result.error.code == 'WORK_AGENT_EXECUTION_FAILED'
    assert 'timeout' in result.error.message


async def test_an_interrupt_gives_the_cli_time_to_confirm_and_raises_when_it_does_not() -> None:
    runtime = SandboxRuntime([{'success': True}, {'success': False, 'stderr': 'still busy'}])
    executor = _executor(runtime=runtime)

    await executor.interrupt(_SESSION)
    with pytest.raises(RuntimeError, match='still busy'):
        await executor.interrupt(_SESSION)

    assert runtime.calls[0]['grace_timeout'] >= 20.0, 'A project Session interrupt must not quarantine a healthy session'
    assert runtime.calls[0]['session_id'] == _CHANNEL


async def test_an_interrupt_not_confirmed_in_time_raises() -> None:
    async def hang() -> dict[str, Any]:
        await asyncio.sleep(1)
        return {'success': True}

    executor = _executor(runtime=SandboxRuntime([hang]), turn_settle_timeout_s=0.05)

    with pytest.raises(RuntimeError, match='not confirmed'):
        await executor.interrupt(_SESSION)


async def test_a_steer_is_asked_once_and_its_outcome_returned() -> None:
    runtime = SandboxRuntime([
        {'success': True, 'steered': True},
        {'success': False, 'steered': False, 'error_code': -32600, 'stderr': 'not ready'},
        RuntimeError('no steer in this image'),
    ])
    executor = _executor(runtime=runtime, trusted_config=_WORKBUDDY)

    assert await executor.steer(_SESSION, prompt='改成 C') == {'steered': True, 'error_code': None, 'stderr': ''}
    assert await executor.steer(_SESSION, prompt='改成 C') == {
        'steered': False,
        'error_code': -32600,
        'stderr': 'not ready',
    }
    refused = await executor.steer(_SESSION, prompt='改成 C')
    assert refused['steered'] is False, 'a runtime image without steering refuses the steer'
    assert [call['prompt'] for call in runtime.calls] == ['改成 C'] * 3
    assert runtime.calls[0]['session_id'] == _CHANNEL


async def test_a_side_question_returns_the_runtime_answer_and_a_failure_as_one() -> None:
    runtime = SandboxRuntime([{'success': True, 'answer': '写到了首页'}, RuntimeError('peer went away')])
    executor = _executor(runtime=runtime, trusted_config=_WORKBUDDY)

    assert await executor.side_question(_SESSION, question='做到哪了？') == {'success': True, 'answer': '写到了首页'}
    failed = await executor.side_question(_SESSION, question='做到哪了？')
    assert failed['success'] is False
    assert failed['stderr'] == 'peer went away'
    assert runtime.calls[0] == {'session_id': _CHANNEL, 'question': '做到哪了？'}


async def test_closing_a_session_cancels_its_host_turn_and_releases_its_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released: list[tuple[str, str]] = []
    runtime = SandboxRuntime()
    started = asyncio.Event()
    binder = RecordingBinder()
    ended: list[Ended] = []

    async def close(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        released.append(('close', kwargs['session_id']))
        return {'success': True}

    monkeypatch.setattr(backend_module, 'call_runtime_agent', close)
    monkeypatch.setattr(backend_module, 'unbind_llm_channel', lambda rt, channel: released.append(('llm', channel)))
    executor = _executor(runtime=runtime, binder=binder, ended=ended)

    async def forever(prompt: str, config: dict[str, Any]) -> TaskResult:
        started.set()
        await asyncio.sleep(10)
        raise AssertionError('closing cancels the turn')

    monkeypatch.setattr(executor, '_invoke_codebuddy', forever)
    await executor.prompt(_SESSION, turn_id='turn-1', prompt='work')
    await started.wait()
    await executor.close_session(_SESSION)

    assert released == [('close', _CHANNEL), ('llm', _CHANNEL)]
    assert binder.unbinds == [_CHANNEL]
    assert ended == [], 'a turn hylive closed is not reported back'


async def test_a_failed_session_close_still_releases_its_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    released: list[tuple[str, str]] = []

    async def failing_close(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError('sandbox is gone')

    monkeypatch.setattr(backend_module, 'call_runtime_agent', failing_close)
    monkeypatch.setattr(backend_module, 'unbind_llm_channel', lambda rt, channel: released.append(('llm', channel)))
    binder = RecordingBinder()
    executor = _executor(binder=binder)

    with pytest.raises(RuntimeError, match='sandbox is gone'):
        await executor.close_session(_SESSION)

    assert released == [('llm', _CHANNEL)]
    assert binder.unbinds == [_CHANNEL]


def _live_input(**fields: Any) -> agent_module.LiveAgentAgentInput:
    return agent_module.LiveAgentAgentInput.model_validate({'prompt': 'build', **fields})


def test_codebuddy_config_passes_through_and_names_no_acp_cli() -> None:
    config = agent_module.project_backend_config(
        _live_input(
            backend_adapter='codebuddy',
            work_agent_configs={'codebuddy': {'skip_permissions': False}, 'workbuddy': {'x': 1}},
        )
    )

    assert config == {'skip_permissions': False, 'llm_client_name': 'external'}


def test_default_adapter_reads_the_workbuddy_config_and_selects_the_workbuddy_cli() -> None:
    config = agent_module.project_backend_config(
        _live_input(
            work_agent_configs={'codebuddy': {'skip_permissions': True}, 'workbuddy': {'skip_permissions': False}},
        )
    )

    assert config == {'skip_permissions': False, 'acp_cli': 'workbuddy', 'llm_client_name': 'external'}


def test_an_acp_cli_that_contradicts_the_adapter_is_rejected() -> None:
    with pytest.raises(ValueError, match='backend_adapter'):
        agent_module.project_backend_config(
            _live_input(backend_adapter='workbuddy', work_agent_configs={'workbuddy': {'acp_cli': 'codebuddy'}})
        )


@pytest.mark.parametrize(('trusted_config', 'expected'), [(None, False), (_WORKBUDDY, True)])
def test_only_workbuddy_steers_answers_btw_and_registers_artifacts_from_acp(
    trusted_config: dict[str, Any] | None, expected: bool
) -> None:
    executor = _executor(trusted_config=trusted_config)

    assert executor.supports_side_question() is expected
    assert executor.supports_steer() is expected
    assert executor.registers_artifacts_from_acp() is expected


class PreloadRuntime:
    """Sandbox runtime that records preload installs and fails the ones listed in ``return_codes``."""

    def __init__(self, return_codes: list[int] | None = None, delay_s: float = 0.0) -> None:
        self.commands: list[tuple[str, dict[str, str] | None]] = []
        self.return_codes = list(return_codes or [])
        self.delay_s = delay_s

    def on(self, event: str, handler: Any) -> None:
        return None

    async def run_command(self, command: str, *, timeout: float, extra_env: dict[str, str] | None = None) -> Any:
        self.commands.append((command, extra_env))
        await asyncio.sleep(self.delay_s)
        code = self.return_codes.pop(0) if self.return_codes else 0
        return SimpleNamespace(return_code=code, stdout='', stderr='npm ERR! network' if code else '', status='done')


async def _turns(executor: AcpSessionExecutor, ended: list[Ended], count: int) -> list[TaskResult]:
    for index in range(count):
        await executor.prompt(_SESSION, turn_id=f'turn-{index}', prompt='research')
        await _until(lambda index=index: len(ended) > index)
    return [result for _, _, result in ended]


@pytest.mark.usefixtures('real_preload_install')
async def test_every_turn_installs_the_preload_once_and_keeps_the_configured_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _patch_codebuddy(monkeypatch, _codebuddy_output(), [{'role': 'assistant', 'content': 'done'}])
    runtime = PreloadRuntime()
    key_env = {'CODEBUDDY_API_KEY': 'codebuddy-key', 'CODEBUDDY_INTERNET_ENVIRONMENT': 'internal'}
    ended: list[Ended] = []
    executor = _executor(runtime=runtime, trusted_config={'extra_env': key_env}, ended=ended)

    results = await _turns(executor, ended, 2)

    assert [result.execution_status for result in results] == ['success', 'success']
    assert len(runtime.commands) == 1, 'The preload is installed once per sandbox'
    command, command_env = runtime.commands[0]
    assert 'undici@6' in command
    assert command_env == {'NODE_OPTIONS': ''}, 'npm must not load the preload it is installing'
    expected = {**key_env, 'NODE_OPTIONS': _PRELOAD_OPTION}
    assert [agent_input.extra_env for agent_input in inputs] == [expected] * 2


async def test_the_preload_goes_before_configured_node_options(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _patch_codebuddy(monkeypatch, _codebuddy_output(), [{'role': 'assistant', 'content': 'done'}])
    ended: list[Ended] = []
    executor = _executor(trusted_config={'extra_env': {'NODE_OPTIONS': '--max-old-space-size=4096'}}, ended=ended)

    await _turns(executor, ended, 1)

    assert inputs[0].extra_env == {'NODE_OPTIONS': f'{_PRELOAD_OPTION} --max-old-space-size=4096'}


async def test_a_workbuddy_turn_proxies_its_children(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _patch_codebuddy(monkeypatch, _codebuddy_output(), [{'role': 'assistant', 'content': 'done'}])
    ended: list[Ended] = []

    await _turns(_executor(trusted_config=_WORKBUDDY, ended=ended), ended, 1)

    assert inputs[0].extra_env['NEXUS_WORKBUDDY_PROXY_CHILDREN'] == '1'


@pytest.mark.usefixtures('real_preload_install')
async def test_concurrent_turns_share_one_preload_install() -> None:
    runtime = PreloadRuntime(delay_s=0.05)
    executor = _executor(runtime=runtime)

    await asyncio.gather(executor._ensure_proxy_preload(), executor._ensure_proxy_preload())

    assert len(runtime.commands) == 1


@pytest.mark.usefixtures('real_preload_install')
async def test_a_failed_preload_fails_the_turn_and_the_next_turn_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _patch_codebuddy(monkeypatch, _codebuddy_output(), [{'role': 'assistant', 'content': 'done'}])
    runtime = PreloadRuntime(return_codes=[1, 0])
    ended: list[Ended] = []
    executor = _executor(runtime=runtime, ended=ended)

    failed, succeeded = await _turns(executor, ended, 2)

    assert failed.execution_status == 'failed'
    assert failed.failure_kind == 'infrastructure'
    assert failed.error is not None
    assert 'proxy preload install failed' in failed.error.message
    assert succeeded.execution_status == 'success'
    assert len(inputs) == 1, 'CodeBuddy must not start without the preload its NODE_OPTIONS requires'


async def test_closing_the_executor_closes_the_sessions_hylive_left_open(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[str] = []
    ended: list[Ended] = []

    async def close(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        closed.append(kwargs['session_id'])
        return {'success': True}

    monkeypatch.setattr(backend_module, 'call_runtime_agent', close)
    monkeypatch.setattr(backend_module, 'unbind_llm_channel', lambda *_: None)
    _patch_codebuddy(monkeypatch, _codebuddy_output(), [{'role': 'assistant', 'content': 'done'}])
    executor = _executor(ended=ended)
    other = SessionRef(session_id='project-2', owner_id='owner-1', working_dir='/workspace')
    await executor.prompt(_SESSION, turn_id='turn-1', prompt='build')
    await executor.prompt(other, turn_id='turn-2', prompt='write')
    await _until(lambda: len(ended) == 2)
    await executor.close_session(other)

    await executor.aclose()

    other_channel = normalize_stream_json_session_id('project-2', namespace='codebuddy-acp-session')
    assert closed == [other_channel, _CHANNEL], 'the session hylive closed is not closed twice'
