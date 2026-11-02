import json

from types import SimpleNamespace

import pytest

from nexus.extensions.agents.installed_common import (
    DEFAULT_NPM_REGISTRY,
    build_npm_env,
    call_runtime_agent,
    classify_command_failure,
    coerce_user_prompt,
    unpack_command_result,
)
from nexus.extensions.agents.installed_common.execution import ComposeExecutionTarget, agent_execution_target
from nexus.runtime.executors.command import CommandRunStatus, RunCommandResponse
from nexus.runtime.models import InvokeRequest
from nexus.runtime.server.services.llm_handler import LLMRequestRouter
from nexus.trace import RemoteTraceContext


def test_coerce_user_prompt_preserves_string():
    assert coerce_user_prompt('write hello') == 'write hello'


def test_coerce_user_prompt_serializes_structured_prompt():
    prompt = [{'type': 'text', 'text': 'write hello'}]

    serialized = coerce_user_prompt(prompt)

    assert json.loads(serialized) == prompt


def test_build_npm_env_defaults_to_tencent_mirror():
    env = build_npm_env()

    assert env['NPM_CONFIG_REGISTRY'] == DEFAULT_NPM_REGISTRY
    assert env['npm_config_registry'] == DEFAULT_NPM_REGISTRY
    # npm env explicitly prepends nix bin for npm/node, but inherited PATH is sanitised
    assert env['PATH'].startswith('/nix/runtime/nix-env/bin')


def test_unpack_command_result_uses_fallback_messages():
    timeout_result = RunCommandResponse(
        status=CommandRunStatus.TimeLimitExceeded,
        stdout='partial',
        stderr='',
    )
    error_result = RunCommandResponse(
        status=CommandRunStatus.Error,
        stdout='',
        stderr='',
    )

    assert unpack_command_result(
        timeout_result,
        timeout_message='timed out',
        error_message='failed',
    ) == ('partial', 'timed out', -1)
    assert unpack_command_result(
        error_result,
        timeout_message='timed out',
        error_message='failed',
    ) == ('', 'failed', -1)


@pytest.mark.parametrize(
    ('status', 'process_failed', 'expected'),
    [
        (CommandRunStatus.Finished, False, (None, None)),
        (CommandRunStatus.Finished, True, ('agent', 'process')),
        (CommandRunStatus.TimeLimitExceeded, True, ('agent', 'process')),
        (CommandRunStatus.Error, True, ('infrastructure', 'runtime_executor')),
    ],
)
def test_classify_command_failure_preserves_executor_status(status, process_failed, expected):
    result = RunCommandResponse(status=status, return_code=1 if process_failed else 0)

    assert classify_command_failure(result, process_failed=process_failed) == expected


@pytest.mark.parametrize(
    'trace_context',
    [None, RemoteTraceContext(trace_id='trace-1', parent_span_id='agent-span-1')],
)
@pytest.mark.asyncio
@pytest.mark.parametrize('disable_invoke_wait_timeout', [False, True])
async def test_call_runtime_agent_builds_request_with_optional_runtime_trace_context(
    monkeypatch, trace_context, disable_invoke_wait_timeout
):
    captured: dict[str, object] = {}

    class FakeRuntime:
        default_venv_id = 'worker-venv-1'

        async def ws_connect(self):
            captured['connected'] = True

        async def invoke(self, request, **kwargs):
            captured['request'] = request
            captured['invoke_kwargs'] = kwargs
            return {'success': True}

    class FakeHandler:
        request_outcomes = {'r1': 'context_length_exceeded', 'r2': None}
        last_request_error_code = None
        request_finish_reasons = {'r1': None, 'r2': 'length'}
        request_unparsed_tool_calls = {'r1': False, 'r2': True}

        def __init__(self, **_kwargs):
            pass

        def register(self):
            captured['registered'] = True

        def unregister(self):
            captured['unregistered'] = True

    runtime = FakeRuntime()

    async def get_runtime():
        return runtime

    ctx = SimpleNamespace(get_runtime=get_runtime, openai_client=object())
    monkeypatch.setattr('nexus.core.require_context', lambda: ctx)
    monkeypatch.setattr('nexus.runtime.server.services.llm_handler.LLMRequestHandler', FakeHandler)

    result = await call_runtime_agent(
        'fake.module::session_turn',
        runtime_trace_context=trace_context,
        disable_invoke_wait_timeout=disable_invoke_wait_timeout,
        session_id='session-1',
        prompt='hello',
    )

    # Per-request LLM outcomes ride back on the result so the caller can tell a call refused on
    # token budget from a real crash, without a run-wide flag that an early refusal would pin.
    assert result == {
        'success': True,
        'llm_request_outcomes': {'r1': 'context_length_exceeded', 'r2': None},
        'last_llm_error_code': None,
        'last_llm_finish_reason': 'length',
        'last_llm_unparsed_tool_call': True,
    }
    assert captured['connected'] is True
    assert captured['registered'] is True
    assert captured['unregistered'] is True
    assert captured['invoke_kwargs'] == {}
    request = captured['request']
    assert isinstance(request, InvokeRequest)
    assert request.trace_context == trace_context
    assert request.disable_invoke_wait_timeout is disable_invoke_wait_timeout
    assert 'disable_invoke_wait_timeout' not in request.model_dump(mode='json')
    assert request.venv_id == 'worker-venv-1'
    _args, kwargs = request.unpack_payload()
    assert kwargs == {'session_id': 'session-1', 'prompt': 'hello'}


@pytest.mark.asyncio
@pytest.mark.parametrize('direct', [False, True])
async def test_call_runtime_agent_injects_selected_execution_target(monkeypatch, direct):
    captured: dict[str, object] = {}

    async def runtime_entry(*, execution_target=None):
        return execution_target

    class FakeRuntime:
        default_venv_id = None

        async def ws_connect(self):
            return None

        async def invoke(self, request):
            captured['request'] = request
            return {'success': True}

    class FakeHandler:
        request_outcomes = {}
        last_request_error_code = None
        request_finish_reasons = {}
        request_unparsed_tool_calls = {}

        def __init__(self, **_kwargs):
            pass

        def register(self):
            pass

        def unregister(self):
            pass

    runtime = FakeRuntime()

    async def get_runtime():
        return runtime

    ctx = SimpleNamespace(get_runtime=get_runtime, openai_client=object())
    monkeypatch.setattr('nexus.core.require_context', lambda: ctx)
    monkeypatch.setattr('nexus.runtime.server.services.llm_handler.LLMRequestHandler', FakeHandler)
    target = ComposeExecutionTarget(
        project_name='task-run-1',
        compose_files=['/harbor/compose/base.yaml'],
        service='main',
        transport_host_dir='/tmp/nxa/123456789abc',
        transport_container_dir='/opt/nexus-agent',
        socket_bridge_host_path='/tmp/nxa/123456789abc/bin/nexus-socket-bridge',
        socket_bridge_container_path='/opt/nexus-agent/bin/nexus-socket-bridge',
    )

    if direct:
        from nexus.extensions.agents.installed_common.execution import DirectExecutionTarget

        target = DirectExecutionTarget(user=1000)

    with agent_execution_target(target):
        await call_runtime_agent(runtime_entry)

    request = captured['request']
    assert isinstance(request, InvokeRequest)
    _args, kwargs = request.unpack_payload()
    assert kwargs['execution_target']['kind'] == ('direct' if direct else 'compose')
    if direct:
        assert kwargs['execution_target']['user'] == 1000
    else:
        assert kwargs['execution_target']['service'] == 'main'


@pytest.mark.asyncio
async def test_call_runtime_agent_rejects_runtime_entry_without_execution_target(monkeypatch):
    async def unsupported_runtime_entry():
        return None

    class FakeRuntime:
        async def ws_connect(self):
            return None

    runtime = FakeRuntime()

    async def get_runtime():
        return runtime

    ctx = SimpleNamespace(get_runtime=get_runtime, openai_client=object())
    monkeypatch.setattr('nexus.core.require_context', lambda: ctx)
    target = ComposeExecutionTarget(
        project_name='task-run-1',
        compose_files=['/harbor/compose/base.yaml'],
        service='main',
        transport_host_dir='/tmp/nxa/123456789abc',
        transport_container_dir='/opt/nexus-agent',
        socket_bridge_host_path='/tmp/nxa/123456789abc/bin/nexus-socket-bridge',
        socket_bridge_container_path='/opt/nexus-agent/bin/nexus-socket-bridge',
    )

    with agent_execution_target(target), pytest.raises(ValueError, match='does not support the selected'):
        await call_runtime_agent(unsupported_runtime_entry)


class _ChannelRuntime:
    default_venv_id = None

    def __init__(self, *, leave_model_call_in_flight: bool = False) -> None:
        self.listeners = {}
        self.leave_model_call_in_flight = leave_model_call_in_flight

    async def ws_connect(self):
        pass

    async def invoke(self, _request):
        if self.leave_model_call_in_flight:
            # The agent ends its turn while one model call it issued is still being answered.
            handler = LLMRequestRouter.for_runtime(self).get('session-1')
            handler._active_requests += 1
            handler._idle_event.clear()
        return {'success': True, 'output': 'turn done'}

    def on(self, event, handler):
        self.listeners[event] = handler

    def off(self, event, handler):
        if self.listeners.get(event) == handler:
            self.listeners.pop(event)


def _use_channel_runtime(monkeypatch, runtime: _ChannelRuntime) -> None:
    async def get_runtime():
        return runtime

    ctx = SimpleNamespace(get_runtime=get_runtime, openai_client=object())
    monkeypatch.setattr('nexus.core.require_context', lambda: ctx)


@pytest.mark.asyncio
async def test_call_runtime_agent_reuses_persistent_channel_handler(monkeypatch):
    runtime = _ChannelRuntime()
    _use_channel_runtime(monkeypatch, runtime)

    await call_runtime_agent('fake.module::turn', channel_id='session-1', persistent_channel=True)
    first_handler = LLMRequestRouter.for_runtime(runtime).get('session-1')
    assert first_handler is not None
    first_handler._thinking_cache.append({'content': 'kept'})

    await call_runtime_agent('fake.module::turn', channel_id='session-1', persistent_channel=True)
    second_handler = LLMRequestRouter.for_runtime(runtime).get('session-1')

    assert second_handler is first_handler
    assert second_handler._thinking_cache == [{'content': 'kept'}]


@pytest.mark.asyncio
async def test_call_runtime_agent_keeps_successful_turn_when_persistent_channel_stays_busy(monkeypatch, caplog):
    runtime = _ChannelRuntime(leave_model_call_in_flight=True)
    _use_channel_runtime(monkeypatch, runtime)
    monkeypatch.setattr('nexus.extensions.agents.installed_common._CHANNEL_IDLE_TIMEOUT', 0.01)

    with caplog.at_level('WARNING'):
        result = await call_runtime_agent('fake.module::turn', channel_id='session-1', persistent_channel=True)

    assert result['success'] is True, 'A model call still in flight must not discard a turn the runtime completed'
    assert result['output'] == 'turn done'
    assert 'session-1' in caplog.text, 'The lingering model call must be visible in the logs'
    LLMRequestRouter.for_runtime(runtime).unbind('session-1')


@pytest.mark.asyncio
async def test_call_runtime_agent_reports_last_llm_call_of_this_turn_on_persistent_channel(monkeypatch):
    class _LengthCutFirstTurnRuntime(_ChannelRuntime):
        turns = 0

        async def invoke(self, _request):
            self.turns += 1
            if self.turns == 1:
                handler = LLMRequestRouter.for_runtime(self).get('session-1')
                handler._record_outcome('r1', None, {'choices': [{'finish_reason': 'length'}]})
            return {'success': True}

    runtime = _LengthCutFirstTurnRuntime()
    _use_channel_runtime(monkeypatch, runtime)

    first = await call_runtime_agent('fake.module::turn', channel_id='session-1', persistent_channel=True)
    second = await call_runtime_agent('fake.module::turn', channel_id='session-1', persistent_channel=True)
    LLMRequestRouter.for_runtime(runtime).unbind('session-1')

    assert first['last_llm_finish_reason'] == 'length'
    assert second['llm_request_outcomes'] == {}
    assert second['last_llm_finish_reason'] is None, 'A turn without model calls must not inherit the previous cut'
    assert second['last_llm_unparsed_tool_call'] is False
