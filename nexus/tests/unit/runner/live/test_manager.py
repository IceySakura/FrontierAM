import asyncio
import re

from types import SimpleNamespace
from typing import Any

import pytest

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    TaskRunInput,
    TaskRunOutput,
    UserInputRequest,
    WithAgentRunner,
    agent_registry,
    task_registry,
)
from nexus.core.agent import WithInputProvider
from nexus.core.context import ExecutionContext, require_context
from nexus.runner.live.manager import LiveRun, LiveRunError, LiveRunManager
from nexus.runner.live.models import (
    CreateRunRequest,
    SubmitTurnRequest,
    request_cache_key,
)
from nexus.runner.single import ModuleObjectConfig, RunHookError, RunHookFailure, RunInput, RunOutput
from nexus.runner.trace import RunTraceProcessor, RunTraceSinkError
from nexus.runtime.providers.base import RuntimeProvider
from nexus.trace.controller import SpanController
from nexus.trace.events import SpanEndEvent, SpanStartEvent, SpanUpdateEvent
from nexus.trace.types import Event


def live_request(**kwargs: Any) -> CreateRunRequest:
    kwargs.setdefault('task_name', 'fake_task')
    return CreateRunRequest(**kwargs)


def cache_live_config(*key_paths: str) -> dict[str, Any]:
    return {
        'cache': {'namespace': 'unit-test', 'key_paths': list(key_paths or ('task_name', 'task_config.user_prompt'))}
    }


def test_live_run_preserves_gateway_trajectory_observers() -> None:
    run_input = RunInput(
        task_name='fake_task',
        gateway_trajectory_observers={
            'langfuse': ModuleObjectConfig(name='langfuse', args={'flush_on_final': False}),
        },
    )

    request = LiveRun.create_request_from_run_input(run_input)
    rebuilt = LiveRun(request)._build_run_input()

    assert request.gateway_trajectory_observers == run_input.gateway_trajectory_observers
    assert rebuilt.gateway_trajectory_observers == run_input.gateway_trajectory_observers


@pytest.mark.parametrize('policy', ['raise', 'warn'])
def test_live_run_preserves_trace_drain_error_policy(policy) -> None:
    run_input = RunInput(task_name='fake_task', runner={'trace_drain_error_policy': policy})
    request = LiveRun.create_request_from_run_input(run_input)
    assert LiveRun(request)._build_run_input().runner.trace_drain_error_policy == policy


async def wait_for_status(run: LiveRun, status: str) -> None:
    for _ in range(100):
        if run.status == status:
            return
        await asyncio.sleep(0.01)
    assert run.status == status


async def wait_for_evicted(manager: LiveRunManager, run_id: str) -> None:
    for _ in range(100):
        if run_id not in manager.runs:
            return
        await asyncio.sleep(0.01)
    assert run_id not in manager.runs


def test_request_cache_key_requires_explicit_live_cache_paths() -> None:
    request = live_request(task_config={'user_prompt': 'same'})

    assert request_cache_key(request) is None


def test_request_cache_key_uses_stable_dot_paths_without_exposing_values() -> None:
    first = live_request(
        run_id='run_one',
        task_config={'user_prompt': 'same', 'trace_id': 'trace-a'},
        runtime_provider_config={'token': 'secret-a'},
        live=cache_live_config('task_name', 'task_config.user_prompt'),
    )
    second = live_request(
        run_id='run_two',
        task_config={'user_prompt': 'same', 'trace_id': 'trace-b'},
        runtime_provider_config={'token': 'secret-b'},
        live=cache_live_config('task_name', 'task_config.user_prompt'),
    )
    different = live_request(
        task_config={'user_prompt': 'different', 'trace_id': 'trace-a'},
        runtime_provider_config={'token': 'secret-a'},
        live=cache_live_config('task_name', 'task_config.user_prompt'),
    )

    cache_key = request_cache_key(first)

    assert cache_key == request_cache_key(second)
    assert cache_key != request_cache_key(different)
    assert re.fullmatch(r'sha256:[0-9a-f]{64}', cache_key or '')
    assert 'secret' not in cache_key
    assert 'same' not in cache_key


def test_request_cache_key_rejects_missing_or_invalid_dot_paths() -> None:
    with pytest.raises(ValueError, match='missing live cache key path'):
        request_cache_key(live_request(live=cache_live_config('task_config.missing')))

    with pytest.raises(ValueError, match='invalid live cache key path'):
        request_cache_key(live_request(live=cache_live_config('task_config.*')))


def test_request_cache_key_namespace_isolates_cache_pools() -> None:
    first = live_request(task_config={'user_prompt': 'same'}, live=cache_live_config())
    second = live_request(
        task_config={'user_prompt': 'same'},
        live={'cache': {'namespace': 'other-unit-test', 'key_paths': ['task_name', 'task_config.user_prompt']}},
    )

    assert request_cache_key(first) != request_cache_key(second)


def test_request_cache_key_disabled_returns_none_even_with_key_paths() -> None:
    request = live_request(
        task_config={'user_prompt': 'same'},
        live={'cache': {'enabled': False, 'namespace': 'unit-test', 'key_paths': ['task_name']}},
    )

    assert request_cache_key(request) is None


@pytest.mark.asyncio
async def test_live_run_manager_returns_before_runtime_ready(monkeypatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def fake_run_single(_run_input, *, hooks=None):
        started.set()
        await release.wait()
        output = RunOutput(task_output={'ok': True})
        await hooks.on_run_output(None, output)
        return output

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)

    manager = LiveRunManager()
    response = await manager.create_run(live_request(runtime_provider_name='fake'))

    assert response.status == 'starting'
    run = manager.get_run(response.run_id)
    assert run.status == 'starting'
    await asyncio.wait_for(started.wait(), timeout=1)

    release.set()
    assert run._task is not None
    await run._task
    assert run.status == 'closed'


@pytest.mark.asyncio
async def test_live_run_manager_persists_closed_run_history(tmp_path) -> None:
    db_path = tmp_path / 'live.sqlite'

    manager = LiveRunManager(storage_path=db_path)
    response = await manager.create_run(
        CreateRunRequest(
            run_id='run_persisted',
            imports=[
                'nexus.extensions.tasks.identity.task',
                'nexus.extensions.agents.main.agent',
                'nexus.extensions.openai_clients.mock',
            ],
            task_name='identity',
            agent_name='main_agent',
            agent_config={'user_prompt': 'finish immediately', 'tools': [], 'max_iterations': 1},
            openai_clients={
                'main': {
                    'name': 'mock',
                    'config': {'client_args': {'responses': ['persisted answer']}},
                }
            },
        )
    )
    run = manager.get_run(response.run_id)
    assert run._task is not None
    await run._task
    await manager.close()

    restored = LiveRunManager(storage_path=db_path)

    infos = await restored.list_run_infos()
    assert [info.run_id for info in infos] == ['run_persisted']
    info = await restored.get_run_info('run_persisted')
    assert info.status == 'closed'
    assert info.events_cursor > 0
    assert info.output is not None
    assert info.output['agent_output']['final_answer'] == 'persisted answer'
    await restored.close()


@pytest.mark.asyncio
async def test_live_run_manager_persists_raw_trace_metadata_at_terminal(monkeypatch, tmp_path) -> None:
    context_ready = asyncio.Event()
    release = asyncio.Event()

    async def fake_run_single(run_input, *, hooks=None):
        await hooks.on_context_ready(ExecutionContext(task_name=run_input.task_name, run_id=run_input.run_id))
        context_ready.set()
        await release.wait()
        output = RunOutput(task_output={'run_id': run_input.run_id})
        await hooks.on_run_output(None, output)
        return output

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)

    db_path = tmp_path / 'live.sqlite'
    manager = LiveRunManager(storage_path=db_path, terminal_run_retention_seconds=-1)
    response = await manager.create_run(live_request(run_id='run_snapshot_flush'))
    run = manager.get_run(response.run_id)

    await asyncio.wait_for(context_ready.wait(), timeout=1)
    assert run.status == 'running'
    assert run.events.latest_id > 0
    assert manager.store is not None
    assert await manager.store.get_trace_metadata(response.run_id) is None

    release.set()
    assert run._task is not None
    await run._task

    assert run.status == 'closed'
    expected_events = run.events.snapshot_after(0)
    assert await manager.store.get_trace_metadata(response.run_id) is not None
    await manager.close()

    restored = LiveRunManager(storage_path=db_path)
    assert await restored.get_run_events_after(response.run_id, 0) == expected_events
    await restored.close()


@pytest.mark.asyncio
async def test_live_run_output_does_not_mark_terminal_until_run_single_returns(monkeypatch) -> None:
    output_appended = asyncio.Event()
    release_return = asyncio.Event()

    async def fake_run_single(run_input, *, hooks=None):
        await hooks.on_context_ready(ExecutionContext(task_name=run_input.task_name, run_id=run_input.run_id))
        output = RunOutput(task_output={'run_id': run_input.run_id})
        await hooks.on_run_output(None, output)
        output_appended.set()
        await release_return.wait()
        return output

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)

    manager = LiveRunManager()
    response = await manager.create_run(live_request(run_id='run_output_before_terminal'))
    run = manager.get_run(response.run_id)

    await asyncio.wait_for(output_appended.wait(), timeout=1)
    assert run.output is not None
    assert run.status == 'running'
    assert [event.type for event in run.events.snapshot_after(0)][-1] == 'run.output'

    release_return.set()
    assert run._task is not None
    await run._task

    assert run.status == 'closed'
    assert [event.type for event in run.events.snapshot_after(0)][-1] == 'run.status'
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('policy', ['raise', 'warn'])
async def test_live_run_trace_failure_closes_session_without_recursive_emit(monkeypatch, policy) -> None:
    from nexus.runner.trace import RunTraceSink

    class FailOnOutputSink(RunTraceSink):
        def __init__(self) -> None:
            self.closed = False

        async def start(self, context) -> None:  # noqa: ANN001
            return None

        async def on_event(self, context, event) -> None:  # noqa: ANN001
            for item in event.event_data.get('events') or []:
                attributes = item.get('attributes') or {}
                if attributes.get('event_type') == 'run.output':
                    raise OSError('output sink failed')

        async def close(self, context) -> None:  # noqa: ANN001
            self.closed = True

    async def fake_run_single(run_input, *, hooks=None):
        await hooks.on_context_ready(ExecutionContext(task_name=run_input.task_name, run_id=run_input.run_id))
        output = RunOutput(task_output={'run_id': run_input.run_id})
        await hooks.on_run_output(None, output)
        return output

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)
    SpanController._instance = None
    controller = SpanController.get_instance_sync()
    processor = RunTraceProcessor()
    controller.add_processor(processor)
    sink = FailOnOutputSink()
    run = LiveRun(
        live_request(run_id='run_trace_failure', runner={'trace_drain_error_policy': policy}),
        trace_backend_sinks={'broken': sink},
        run_trace_processor=processor,
    )

    await run.start(wait_until_ready=False)
    assert run._task is not None
    await asyncio.wait_for(run._task, timeout=1)

    assert run.status == 'failed'
    assert sink.closed is True
    with pytest.raises(RunTraceSinkError, match='output sink failed'):
        await run.wait_trace_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize('policy', [None, 'raise', 'warn'])
@pytest.mark.parametrize('failure_stage', ['run_output', 'context_closing'])
async def test_live_run_trace_drain_policy_controls_terminal_status(monkeypatch, caplog, policy, failure_stage) -> None:
    from nexus.runner.single import _invoke_post_run_hook

    current_stage = None

    class Runtime:
        async def drain_trace(self):
            if current_stage == failure_stage:
                raise RuntimeError('trace transport disconnected')

    runtime = Runtime()
    context = SimpleNamespace(runtime_provider=SimpleNamespace(get_active_runtimes=lambda: {'main': runtime}))
    output = RunOutput(task_output={'archive': 'project.tar.gz'})

    async def fake_run_single(_run_input, *, hooks=None):
        nonlocal current_stage
        current_stage = 'run_output'
        await _invoke_post_run_hook(hooks, current_stage, hooks.on_run_output, context, output)
        current_stage = 'context_closing'
        await _invoke_post_run_hook(hooks, current_stage, hooks.on_context_closing, context)
        return output

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)
    runner = {'trace_drain_error_policy': policy} if policy is not None else {}
    run = LiveRun(live_request(run_id='run_drain_policy', runner=runner))
    await run.start(wait_until_ready=False)
    await asyncio.wait_for(run._task, timeout=1)

    if policy == 'raise':
        assert run.status == 'failed'
        assert isinstance(run._terminal_trace_error, RunHookError)
        return

    assert run.status == 'closed'
    assert run.output == output
    assert run._terminal_trace_error is None
    await run.wait_trace_closed()
    outputs = [event for event in run.events.snapshot_after(0) if event.type == 'run.output']
    assert len(outputs) == 1
    assert outputs[0].data['output'] == output.model_dump(mode='json')
    assert 'run_id=run_drain_policy runtime=main error_type=RuntimeError' in caplog.text


@pytest.mark.asyncio
async def test_live_run_drain_hook_failure_skips_output_fallback(monkeypatch) -> None:
    async def fake_run_single(_run_input, *, hooks=None):
        hooks.hook_failures.append(
            RunHookFailure(stage='context_closing', error=RuntimeError('final trace drain failed'))
        )
        return RunOutput(task_output={'ok': True})

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)
    run = LiveRun(live_request(run_id='run_drain_failure'))

    await run.start(wait_until_ready=False)
    assert run._task is not None
    await asyncio.wait_for(run._task, timeout=1)

    assert run.status == 'failed'
    assert run.output is None
    assert isinstance(run._terminal_trace_error, RunHookError)
    assert not any(event.type == 'run.output' for event in run.events.snapshot_after(0))


@pytest.mark.asyncio
async def test_live_run_drain_hook_failure_emits_terminal_status_with_cause(monkeypatch) -> None:
    # Regression: a trace-drain failure at teardown (e.g. runtime Socket.IO disconnect) must
    # surface the real cause via a terminal run.status=failed event, instead of leaving the
    # trace without a terminal status. Previously the driver reported this as the opaque
    # ResultPersistenceFailed / "run trace closed without terminal run.status", hiding the
    # underlying runtime disconnect.
    async def fake_run_single(_run_input, *, hooks=None):
        hooks.hook_failures.append(
            RunHookFailure(
                stage='context_closing',
                error=RuntimeError('runtime Socket.IO disconnected before trace drain'),
            )
        )
        return RunOutput(task_output={'ok': True})

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)
    run = LiveRun(live_request(run_id='run_drain_cause'))

    await run.start(wait_until_ready=False)
    assert run._task is not None
    await asyncio.wait_for(run._task, timeout=1)

    assert run.status == 'failed'
    assert run.output is None
    assert isinstance(run._terminal_trace_error, RunHookError)

    status_events = [event for event in run.events.snapshot_after(0) if event.type == 'run.status']
    assert status_events, 'expected a terminal run.status event carrying the failure cause'
    terminal = status_events[-1]
    assert terminal.data.get('status') == 'failed'
    assert 'Socket.IO disconnected before trace drain' in (terminal.data.get('error') or '')


@pytest.mark.asyncio
async def test_live_run_kill_waits_for_cancel_cleanup_instead_of_using_an_internal_deadline(monkeypatch) -> None:
    from nexus.runner.trace import RunTraceSink

    class NoopSink(RunTraceSink):
        async def start(self, context) -> None:  # noqa: ANN001
            return None

        async def on_event(self, context, event) -> None:  # noqa: ANN001
            return None

        async def close(self, context) -> None:  # noqa: ANN001
            return None

    started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def fake_run_single(_run_input, *, hooks=None):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await release_cleanup.wait()

    async def reject_internal_wait_for(*_args, **_kwargs):
        raise AssertionError('LiveRun.kill must use the caller stop budget')

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)
    run = LiveRun(live_request(run_id='run_kill_cleanup'), trace_backend_sinks={'noop': NoopSink()})
    await run.start(wait_until_ready=False)
    await started.wait()
    monkeypatch.setattr('nexus.runner.live.manager.asyncio.wait_for', reject_internal_wait_for)

    kill_task = asyncio.create_task(run.kill())
    await cleanup_started.wait()
    await asyncio.sleep(0)
    assert kill_task.done() is False

    release_cleanup.set()
    await kill_task
    assert run.status == 'killed'


@pytest.mark.asyncio
async def test_live_run_manager_evicts_terminal_runs_after_retention(monkeypatch, tmp_path) -> None:
    async def fake_run_single(run_input, *, hooks=None):
        output = RunOutput(task_output={'run_id': run_input.run_id})
        await hooks.on_run_output(None, output)
        return output

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)

    manager = LiveRunManager(storage_path=tmp_path / 'live.sqlite', terminal_run_retention_seconds=0.01)
    response = await manager.create_run(live_request(run_id='run_evict_terminal'))
    run = manager.get_run(response.run_id)
    assert run._task is not None
    await run._task
    assert run.status == 'closed'

    await wait_for_evicted(manager, response.run_id)

    with pytest.raises(LiveRunError, match='run not found'):
        manager.get_run(response.run_id)
    info = await manager.get_run_info(response.run_id)
    assert info.status == 'closed'
    assert info.output is not None
    assert info.output['task_output']['run_id'] == response.run_id
    events = await manager.get_run_events_after(response.run_id, 0)
    assert [event.type for event in events][-2:] == ['run.output', 'run.status']
    assert events[-1].data['status'] == 'closed'
    await manager.close()


@pytest.mark.asyncio
async def test_live_run_sqlite_span_processor_does_not_pickle_manager_callbacks(tmp_path) -> None:
    class HookedAgentInput(AgentRunInput, WithInputProvider):
        pass

    class HookedAgentOutput(AgentRunOutput):
        value: str

    class HookedTaskInput(TaskRunInput, WithAgentRunner):
        pass

    class HookedTaskOutput(TaskRunOutput):
        value: str

    agent_name = 'test_live_sqlite_hooked_agent'
    task_name = 'test_live_sqlite_hooked_task'

    @agent_registry.register(agent_name)
    async def hooked_agent(_agent_input: HookedAgentInput) -> HookedAgentOutput:
        return HookedAgentOutput(value='ok')

    @task_registry.register(task_name)
    async def hooked_task(task_input: HookedTaskInput) -> HookedTaskOutput:
        output = await task_input.run_agent()
        return HookedTaskOutput(value=output.value)

    manager = LiveRunManager(storage_path=tmp_path / 'live.sqlite')
    await manager.start()
    try:
        live_run = await manager._register_new_run(
            live_request(run_id='run_sqlite_span_pickle', task_name=task_name, agent_name=agent_name),
            cache_on_first_idle=False,
        )

        await live_run.start(wait_until_ready=False)
        assert live_run._task is not None
        await live_run._task

        assert live_run.output is not None
        assert live_run.output.success is True
        assert live_run.output.task_output.value == 'ok'
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_live_run_manager_keeps_terminal_runs_without_sqlite_store(monkeypatch) -> None:
    async def fake_run_single(run_input, *, hooks=None):
        output = RunOutput(task_output={'run_id': run_input.run_id})
        await hooks.on_run_output(None, output)
        return output

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)

    manager = LiveRunManager(terminal_run_retention_seconds=0)
    response = await manager.create_run(live_request(run_id='run_memory_only_terminal'))
    run = manager.get_run(response.run_id)
    assert run._task is not None
    await run._task
    await asyncio.sleep(0)

    assert manager.get_run(response.run_id).status == 'closed'
    await manager.close()


@pytest.mark.asyncio
async def test_live_run_manager_rejects_duplicate_before_constructing_trace_state() -> None:
    manager = LiveRunManager()
    first = await manager._register_new_run(
        live_request(run_id='duplicate-run'),
        cache_on_first_idle=False,
    )
    await first._trace_session.start()
    original_stream = first.events

    with pytest.raises(LiveRunError, match='run already exists'):
        await manager._register_new_run(
            live_request(run_id='duplicate-run'),
            cache_on_first_idle=False,
        )

    assert first.events is original_stream
    await manager.close()


@pytest.mark.asyncio
async def test_live_run_turns_feed_get_user_input(monkeypatch) -> None:
    prompts: list[str | None] = []

    async def fake_run_single(run_input, *, hooks=None):
        ctx = ExecutionContext(task_name=run_input.task_name, run_id=run_input.run_id)
        async with ctx:
            await hooks.on_context_ready(ctx)
            get_user_input = hooks.agent_input_overrides['get_user_input']
            first_prompt = await get_user_input(
                UserInputRequest(source='hook', hint='ready for first', session_id='session-1')
            )
            prompts.append(first_prompt)
            next_prompt = await get_user_input(
                UserInputRequest(source='turn', hint='ready for next', session_id='session-1')
            )
            prompts.append(next_prompt)
            output = RunOutput(
                task_output={
                    'success': True,
                    'output': 'done',
                    'stderr': '',
                    'exit_code': 0,
                    'execution_time': 0.1,
                    'session_id': 'session-1',
                    'turns': 2,
                    'ended_by_user': True,
                    'reached_max_turns': False,
                    'events': [],
                }
            )
            await hooks.on_run_output(ctx, output)
            return output

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)

    run = LiveRun(live_request(runtime_provider_name='fake'))
    await run.start()

    await wait_for_status(run, 'idle')
    assert run.status == 'idle'

    first = await run.submit_turn(SubmitTurnRequest(text='first'))
    assert first.turn_id == 'turn_1'

    for _ in range(50):
        if run.status == 'idle':
            break
        await asyncio.sleep(0.01)
    assert run.status == 'idle'

    second = await run.submit_turn(SubmitTurnRequest(text='second'))
    assert second.turn_id == 'turn_2'

    assert run._task is not None
    await run._task

    assert prompts == ['first', 'second']
    assert run.status == 'closed'
    assert [event.type for event in run.events.snapshot_after(0)].count('input.required') == 2
    assert run.get_info().output is not None
    assert run.get_info().output['task_output']['output'] == 'done'

    event_count = len(run.events.snapshot_after(0))
    await run.kill()
    assert len(run.events.snapshot_after(0)) == event_count


@pytest.mark.asyncio
async def test_live_run_task_has_execution_context(monkeypatch) -> None:
    async def fake_run_single(run_input, *, hooks=None):
        ctx = ExecutionContext(task_name=run_input.task_name, run_id=run_input.run_id)
        async with ctx:
            await hooks.on_context_ready(ctx)
            assert require_context() is run.ctx
            output = RunOutput(
                task_output={
                    'success': True,
                    'output': 'done',
                    'stderr': '',
                    'exit_code': 0,
                    'execution_time': 0.1,
                    'session_id': 'session-1',
                    'turns': 1,
                    'ended_by_user': False,
                    'reached_max_turns': False,
                    'events': [],
                }
            )
            await hooks.on_run_output(ctx, output)
            return output

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)

    run = LiveRun(live_request(runtime_provider_name='fake', task_config={'user_prompt': 'first'}))

    await run.start()

    assert run._task is not None
    await run._task

    assert run.status == 'closed'


@pytest.mark.asyncio
async def test_live_run_starts_from_task_user_prompt(monkeypatch) -> None:
    prompts: list[str] = []
    subscribed_events: list[str] = []
    trace_drains = 0

    class FakeRuntime:
        base_url = 'http://runtime.test'

        async def ws_connect(self):
            return None

        async def drain_trace(self):
            nonlocal trace_drains
            trace_drains += 1

        def on(self, event_name, *_args):
            subscribed_events.append(event_name)
            return lambda: None

    class FakeProvider(RuntimeProvider):
        async def acquire(self, name: str, **_params: Any) -> Any:
            runtime = FakeRuntime()
            self._active_runtimes[name] = runtime
            return runtime

        async def _release_impl(self, name: str) -> None:
            self._active_runtimes.pop(name, None)

    async def fake_run_single(run_input, *, hooks=None):
        prompts.append(run_input.task_config['user_prompt'])
        ctx = ExecutionContext(
            runtime_provider=FakeProvider(),
            runtime_configs={},
            task_name=run_input.task_name,
            run_id=run_input.run_id,
        )
        async with ctx:
            await hooks.on_context_ready(ctx)
            output = RunOutput(
                task_output={
                    'success': True,
                    'output': 'done',
                    'stderr': '',
                    'exit_code': 0,
                    'execution_time': 0.1,
                    'session_id': 'session-1',
                    'turns': 1,
                    'ended_by_user': False,
                    'reached_max_turns': False,
                    'events': [],
                }
            )
            await hooks.on_run_output(ctx, output)
            await hooks.on_context_closing(ctx)
            return output

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)

    run = LiveRun(
        live_request(
            runtime_provider_name='fake',
            task_config={'user_prompt': 'start now'},
        )
    )

    await run.start()

    assert run._task is not None
    await run._task
    assert prompts == ['start now']
    assert run.runtime_base_url == 'http://runtime.test'
    assert subscribed_events == ['agent_event']
    # run.output, context-closing hook, then ExecutionContext cleanup.
    assert trace_drains == 3
    assert all(event.type != 'turn.submitted' for event in run.events.snapshot_after(0))


@pytest.mark.asyncio
async def test_live_span_processor_relays_openai_completion_start() -> None:
    run = LiveRun(live_request(runtime_provider_name='fake'))
    controller = SpanController()
    controller.add_processor(run._trace_processor)
    await run._trace_session.start()

    controller.add_event(
        SpanStartEvent(
            span_id='span-1',
            trace_id='trace-1',
            parent_span_id='',
            name='openai_completion',
            time_unix_nano=1,
            attributes={
                '_run_id': run.run_id,
                '_run_span_id': run._trace_session.run_span_id,
                '_run_kind': 'live',
                'inputs': {
                    'messages': [
                        {'role': 'system', 'content': 'sys'},
                        {'role': 'user', 'content': 'first user prompt'},
                    ]
                },
            },
        )
    )

    events = await run.events.wait_after(0, timeout=1)

    assert events[-1].type == 'trace.event'
    assert events[-1].data['event']['span_name'] == 'openai_completion'
    assert events[-1].data['event']['event']['attributes']['inputs']['messages'][-1]['content'] == 'first user prompt'
    await run._trace_session.close()


@pytest.mark.asyncio
async def test_live_span_processor_relays_openai_completion_end_outputs() -> None:
    run = LiveRun(live_request(runtime_provider_name='fake'))
    controller = SpanController()
    controller.add_processor(run._trace_processor)
    await run._trace_session.start()
    controller.add_event(
        SpanStartEvent(
            span_id='span-1',
            trace_id='trace-1',
            parent_span_id='',
            name='openai_completion',
            time_unix_nano=1,
            attributes={
                '_run_id': run.run_id,
                '_run_span_id': run._trace_session.run_span_id,
                '_run_kind': 'live',
                'outputs': {
                    'choices': [
                        {
                            'message': {
                                'role': 'assistant',
                                'content': 'done',
                            }
                        }
                    ]
                },
            },
        )
    )
    controller.add_event(SpanEndEvent(span_id='span-1', time_unix_nano=2))

    events = await run.events.wait_after(0, timeout=1)
    end_events = [event for event in events if event.data['event']['event'].get('type') == 'END']

    assert end_events[-1].type == 'trace.event'
    assert end_events[-1].data['event']['event']['type'] == 'END'
    assert end_events[-1].data['event']['event']['attributes']['outputs']['choices'][0]['message']['content'] == 'done'
    await run._trace_session.close()


@pytest.mark.asyncio
async def test_live_span_processor_relays_span_update_events() -> None:
    run = LiveRun(live_request(runtime_provider_name='fake'))
    controller = SpanController()
    controller.add_processor(run._trace_processor)
    await run._trace_session.start()
    controller.add_event(
        SpanStartEvent(
            span_id='span-1',
            trace_id='trace-1',
            parent_span_id='',
            name='openai_completion',
            time_unix_nano=1,
            attributes={'_run_id': run.run_id, '_run_span_id': run._trace_session.run_span_id, '_run_kind': 'live'},
        )
    )
    controller.add_event(
        SpanUpdateEvent(
            span_id='span-1',
            events=[
                Event(
                    time_unix_nano=2,
                    name='response.output_text.delta',
                    attributes={'delta': 'piece'},
                )
            ],
        )
    )

    events = await run.events.wait_after(0, timeout=1)
    update_events = [
        event
        for event in events
        if event.data['event']['event'].get('name') == 'response.output_text.delta'
        or event.data['event']['event'].get('type') == 'response.output_text.delta'
    ]

    assert update_events[-1].type == 'trace.event'
    assert update_events[-1].data['event']['event']['attributes']['delta'] == 'piece'
    await run._trace_session.close()


@pytest.mark.asyncio
async def test_live_run_legacy_turn_api_no_longer_starts_task_directly(monkeypatch) -> None:
    async def fake_run_single(_run_input, *, hooks=None):
        await hooks.on_context_ready(ExecutionContext(task_name='identity'))
        prompt = await hooks.agent_input_overrides['get_user_input']('ready')
        output = RunOutput(task_output={'prompt': prompt})
        await hooks.on_run_output(None, output)
        return output

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)

    run = LiveRun(live_request(runtime_provider_name='fake'))
    await run.start()
    await wait_for_status(run, 'idle')
    assert run.status == 'idle'

    await run.submit_turn(SubmitTurnRequest(text='first'))
    assert run._task is not None
    await run._task
    assert run.get_info().output['task_output']['prompt'] == 'first'


@pytest.mark.asyncio
async def test_live_run_manager_creates_cached_run_at_first_input_boundary(monkeypatch) -> None:
    async def fake_run_single(run_input, *, hooks=None):
        await hooks.on_context_ready(ExecutionContext(task_name=run_input.task_name, run_id=run_input.run_id))
        prompt = await hooks.agent_input_overrides['get_user_input']('ready')
        output = RunOutput(task_output={'prompt': prompt})
        await hooks.on_run_output(None, output)
        return output

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)

    manager = LiveRunManager()
    response = await manager.create_cache_run(
        live_request(task_config={'user_prompt': 'warm'}, live=cache_live_config())
    )
    run = manager.get_run(response.run_id)

    assert response.status == 'starting'
    assert response.cache_key == run.cache_key
    await wait_for_status(run, 'cached')
    assert run.get_info().cache_key == response.cache_key
    with pytest.raises(LiveRunError, match='not idle'):
        await run.submit_turn(SubmitTurnRequest(text='should wait for reuse'))

    await manager.close()


@pytest.mark.asyncio
async def test_live_run_manager_reuses_cached_runs_fifo(monkeypatch) -> None:
    started_run_ids: list[str] = []

    async def fake_run_single(run_input, *, hooks=None):
        started_run_ids.append(run_input.run_id)
        await hooks.on_context_ready(ExecutionContext(task_name=run_input.task_name, run_id=run_input.run_id))
        prompt = await hooks.agent_input_overrides['get_user_input']('ready')
        output = RunOutput(task_output={'prompt': prompt})
        await hooks.on_run_output(None, output)
        return output

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)

    manager = LiveRunManager()
    first = await manager.create_cache_run(live_request(task_config={'user_prompt': 'same'}, live=cache_live_config()))
    second = await manager.create_cache_run(live_request(task_config={'user_prompt': 'same'}, live=cache_live_config()))
    await wait_for_status(manager.get_run(first.run_id), 'cached')
    await wait_for_status(manager.get_run(second.run_id), 'cached')

    reused_first = await manager.create_run(live_request(task_config={'user_prompt': 'same'}, live=cache_live_config()))
    reused_second = await manager.create_run(
        live_request(task_config={'user_prompt': 'same'}, live=cache_live_config())
    )

    assert reused_first.run_id == first.run_id
    assert reused_first.status == 'idle'
    assert reused_first.reused_cache is True
    assert reused_first.cache_key == first.cache_key
    assert reused_second.run_id == second.run_id
    assert reused_second.status == 'idle'
    assert reused_second.reused_cache is True
    assert reused_second.cache_key == second.cache_key
    assert started_run_ids == [first.run_id, second.run_id]

    for reused in (reused_first, reused_second):
        run = manager.get_run(reused.run_id)
        await run.submit_turn(SubmitTurnRequest(text='go'))
        assert run._task is not None
        await run._task
        assert run.get_info().output['task_output']['prompt'] == 'go'


@pytest.mark.asyncio
async def test_live_run_manager_does_not_reuse_cache_for_different_config(monkeypatch) -> None:
    started_count = 0

    async def fake_run_single(run_input, *, hooks=None):
        nonlocal started_count
        started_count += 1
        await hooks.on_context_ready(ExecutionContext(task_name=run_input.task_name, run_id=run_input.run_id))
        await hooks.agent_input_overrides['get_user_input']('ready')

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)

    manager = LiveRunManager()
    cached = await manager.create_cache_run(
        live_request(task_config={'user_prompt': 'first'}, live=cache_live_config())
    )
    await wait_for_status(manager.get_run(cached.run_id), 'cached')

    fresh = await manager.create_run(live_request(task_config={'user_prompt': 'second'}, live=cache_live_config()))

    assert fresh.run_id != cached.run_id
    assert fresh.reused_cache is False
    await wait_for_status(manager.get_run(fresh.run_id), 'idle')
    assert started_count == 2
    await manager.close()


@pytest.mark.asyncio
async def test_live_run_manager_explicit_run_id_skips_cache_reuse(monkeypatch) -> None:
    started_count = 0

    async def fake_run_single(run_input, *, hooks=None):
        nonlocal started_count
        started_count += 1
        await hooks.on_context_ready(ExecutionContext(task_name=run_input.task_name, run_id=run_input.run_id))
        await hooks.agent_input_overrides['get_user_input']('ready')

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)

    manager = LiveRunManager()
    cached = await manager.create_cache_run(live_request(task_config={'user_prompt': 'same'}, live=cache_live_config()))
    await wait_for_status(manager.get_run(cached.run_id), 'cached')

    fresh = await manager.create_run(
        live_request(run_id='run_explicit', task_config={'user_prompt': 'same'}, live=cache_live_config())
    )

    assert fresh.run_id == 'run_explicit'
    assert fresh.reused_cache is False
    await wait_for_status(manager.get_run(fresh.run_id), 'idle')
    assert started_count == 2
    await manager.close()


@pytest.mark.asyncio
async def test_live_run_manager_cache_key_uses_resolved_openai_config(monkeypatch, tmp_path) -> None:
    started_count = 0
    config = tmp_path / 'openai.yaml'
    config.write_text(
        """
task_name: identity
openai_clients:
  main:
    name: mock
    config:
      client_args:
        responses:
          - cached answer
""",
        encoding='utf-8',
    )

    async def fake_run_single(run_input, *, hooks=None):
        nonlocal started_count
        started_count += 1
        await hooks.on_context_ready(ExecutionContext(task_name=run_input.task_name, run_id=run_input.run_id))
        await hooks.agent_input_overrides['get_user_input']('ready')

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)

    manager = LiveRunManager()
    cached = await manager.create_cache_run(
        live_request(openai_config_path=str(config), live=cache_live_config('task_name', 'openai_clients.main.name'))
    )
    await wait_for_status(manager.get_run(cached.run_id), 'cached')

    reused = await manager.create_run(
        live_request(
            openai_clients={
                'main': {
                    'name': 'mock',
                    'config': {'client_args': {'responses': ['cached answer']}},
                }
            },
            live=cache_live_config('task_name', 'openai_clients.main.name'),
        )
    )

    assert reused.run_id == cached.run_id
    assert reused.reused_cache is True
    assert started_count == 1
    await manager.close()


@pytest.mark.asyncio
async def test_live_run_manager_rejects_cache_run_without_key_paths() -> None:
    manager = LiveRunManager()

    with pytest.raises(LiveRunError, match='live.cache.key_paths'):
        await manager.create_cache_run(live_request(task_config={'user_prompt': 'same'}))

    await manager.close()


@pytest.mark.asyncio
async def test_live_run_manager_does_not_reuse_without_key_paths(monkeypatch) -> None:
    started_count = 0

    async def fake_run_single(run_input, *, hooks=None):
        nonlocal started_count
        started_count += 1
        await hooks.on_context_ready(ExecutionContext(task_name=run_input.task_name, run_id=run_input.run_id))
        await hooks.agent_input_overrides['get_user_input']('ready')

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)

    manager = LiveRunManager()
    first = await manager.create_run(live_request(task_config={'user_prompt': 'same'}))
    second = await manager.create_run(live_request(task_config={'user_prompt': 'same'}))

    assert first.cache_key is None
    assert second.cache_key is None
    assert first.reused_cache is False
    assert second.reused_cache is False
    assert first.run_id != second.run_id
    await wait_for_status(manager.get_run(first.run_id), 'idle')
    await wait_for_status(manager.get_run(second.run_id), 'idle')
    assert started_count == 2
    await manager.close()


@pytest.mark.asyncio
async def test_live_run_empty_turn_finishes_user_input(monkeypatch) -> None:
    async def fake_run_single(run_input, *, hooks=None):
        ctx = ExecutionContext(task_name=run_input.task_name, run_id=run_input.run_id)
        async with ctx:
            await hooks.on_context_ready(ctx)
            prompt = await hooks.agent_input_overrides['get_user_input']('ready')
            output = RunOutput(task_output={'prompt': prompt})
            await hooks.on_run_output(ctx, output)
            return output

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)

    run = LiveRun(live_request(runtime_provider_name='fake'))
    await run.start()
    await wait_for_status(run, 'idle')
    assert run.status == 'idle'

    await run.submit_turn(SubmitTurnRequest(text=''))
    assert run._task is not None
    await run._task

    assert run.status == 'closed'
    assert run.get_info().output['task_output']['prompt'] is None
    assert not any(
        event.type == 'turn.started' and event.data.get('turn_id') == 'turn_1' for event in run.events.snapshot_after(0)
    )


@pytest.mark.asyncio
async def test_live_run_kill_cancels_running_task(monkeypatch) -> None:
    cancelled = asyncio.Event()

    async def fake_run_single(run_input, *, hooks=None):
        ctx = ExecutionContext(task_name=run_input.task_name, run_id=run_input.run_id)
        async with ctx:
            await hooks.on_context_ready(ctx)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)

    run = LiveRun(live_request(runtime_provider_name='fake', task_config={'user_prompt': 'start'}))
    await run.start()

    assert run.status == 'running'
    await run.kill()

    assert cancelled.is_set()
    assert run.status == 'killed'
    assert any(event.type == 'run.killed' for event in run.events.snapshot_after(0))
    assert run.get_info().output is None


@pytest.mark.asyncio
async def test_live_run_kill_waits_for_terminal_trace_close(monkeypatch) -> None:
    run = LiveRun(live_request(runtime_provider_name='fake'))
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    close_finished = asyncio.Event()

    async def delayed_close() -> None:
        close_started.set()
        await allow_close.wait()
        close_finished.set()

    monkeypatch.setattr(run._trace_session, 'close', delayed_close)
    run._closed = True
    run.status = 'closed'

    kill_task = asyncio.create_task(run.kill())
    await close_started.wait()
    assert kill_task.done() is False
    assert close_finished.is_set() is False

    allow_close.set()
    await kill_task

    assert close_finished.is_set() is True


@pytest.mark.asyncio
async def test_live_run_terminal_trace_failure_is_observable_and_marks_run_failed(monkeypatch) -> None:
    terminal_statuses: list[str] = []

    async def on_terminal(run: LiveRun) -> None:
        terminal_statuses.append(run.status)

    run = LiveRun(live_request(runtime_provider_name='fake'), on_terminal=on_terminal)
    await run._trace_session.start()
    original_close = run._trace_session.close

    async def failing_close() -> None:
        await original_close()
        raise OSError('trace close failed')

    monkeypatch.setattr(run._trace_session, 'close', failing_close)

    with pytest.raises(OSError, match='trace close failed'):
        await run._set_status('closed')
    with pytest.raises(OSError, match='trace close failed'):
        await run.wait_trace_closed()

    assert run.status == 'failed'
    assert run.error == 'OSError: trace close failed'
    assert terminal_statuses == ['failed']


@pytest.mark.asyncio
async def test_live_run_relays_generic_agent_events() -> None:
    run = LiveRun(live_request(runtime_provider_name='fake'))
    await run._trace_session.start()

    await run._handle_runtime_agent_event({'type': 'assistant', 'message': {'content': 'hello'}})

    events = await run.events.wait_after(0, timeout=1)
    assert events[-1].type == 'agent.event'
    assert events[-1].data['event']['source'] == 'agent_event'
    assert events[-1].data['event']['message']['content'] == 'hello'

    await run._set_status('closed')
    cursor = run.events.latest_id
    await run._handle_runtime_agent_event({'type': 'late'})
    assert await run.events.wait_after(cursor, timeout=0.01) == []


@pytest.mark.asyncio
async def test_live_run_terminal_status_waits_for_inflight_runtime_event(monkeypatch) -> None:
    run = LiveRun(live_request(runtime_provider_name='fake'))
    await run._trace_session.start()
    event_started = asyncio.Event()
    allow_event = asyncio.Event()
    original_emit = run._emit

    async def delayed_emit(event_type: str, **data: Any) -> None:
        if event_type == 'agent.event':
            event_started.set()
            await allow_event.wait()
        await original_emit(event_type, **data)

    monkeypatch.setattr(run, '_emit', delayed_emit)
    agent_event = asyncio.create_task(run._handle_runtime_agent_event({'type': 'assistant'}))
    await event_started.wait()
    terminal = asyncio.create_task(run._set_status('closed'))
    await asyncio.sleep(0)

    assert terminal.done() is False

    allow_event.set()
    await agent_event
    await terminal

    assert [event.type for event in run.events.snapshot_after(0)] == ['agent.event', 'run.status']


def test_live_run_loads_openai_clients_from_config_file(tmp_path) -> None:
    config = tmp_path / 'openclaw.yaml'
    config.write_text(
        """
task_name: identity
runtime_provider_name: docker
openai_clients:
  main:
    name: litellm
    config:
      client_args:
        api_key: secret-value
        base_url: http://model-service.test
      request_args:
        model: anthropic.claude-sonnet-4-6
""",
        encoding='utf-8',
    )
    run = LiveRun(live_request(runtime_provider_name='fake', openai_config_path=str(config)))

    run._load_openai_config_file()

    assert run.request.openai_clients['main']['name'] == 'litellm'
    assert run.request.openai_clients['main']['config']['request_args']['model'] == 'anthropic.claude-sonnet-4-6'


def test_live_run_deep_merges_openai_clients_from_config_file(tmp_path) -> None:
    config = tmp_path / 'openclaw.yaml'
    config.write_text(
        """
task_name: identity
runtime_provider_name: docker
openai_clients:
  main:
    name: litellm
    config:
      client_args:
        api_key: secret-value
        base_url: http://model-service.test
      request_args:
        model: anthropic.claude-sonnet-4-6
""",
        encoding='utf-8',
    )
    run = LiveRun(
        live_request(
            runtime_provider_name='fake',
            openai_config_path=str(config),
            openai_clients={
                'main': {
                    'config': {
                        'request_args': {'model': 'openai/anthropic.claude-sonnet-4-6'},
                    }
                }
            },
        )
    )

    run._load_openai_config_file()

    client_config = run.request.openai_clients['main']['config']
    assert client_config['client_args']['api_key'] == 'secret-value'
    assert client_config['client_args']['base_url'] == 'http://model-service.test'
    assert client_config['request_args']['model'] == 'openai/anthropic.claude-sonnet-4-6'


def test_live_run_runtime_infos_use_run_scoped_proxy_urls() -> None:
    class FakeRuntime:
        base_url = 'http://runtime.test'

    class FakeProvider:
        _runtime_ids = {'main': 'nexus-runtime-123'}

        def get_active_runtimes(self):
            return {'main': runtime}

    runtime = FakeRuntime()
    run = LiveRun(live_request(runtime_provider_name='fake', live={'working_dir': '/workspace'}))
    run.runtime_provider = FakeProvider()  # type: ignore[assignment]
    run.status = 'idle'

    info = run.get_info()

    assert info.runtimes['main'].base_url == 'http://runtime.test'
    assert info.runtimes['main'].runtime_id == 'nexus-runtime-123'
    assert info.runtimes['main'].proxy_url == f'/v1/runs/{run.run_id}/runtimes/main/proxy'
    assert info.runtimes['main'].debug_gui_url == f'/v1/runs/{run.run_id}/runtimes/main/proxy/debug/gui'
    assert run.get_runtime('main') is runtime


def test_live_run_runtime_infos_infer_gongfeng_runtime_id_from_base_url() -> None:
    class FakeRuntime:
        base_url = 'http://nexus-abc123.45951.hyrl-sandbox.prod.woa.com:18888'

    runtime = FakeRuntime()
    run = LiveRun(live_request(runtime_provider_name='fake', live={'working_dir': '/workspace'}))
    run.runtime = runtime  # type: ignore[assignment]
    run.status = 'idle'

    info = run.get_info()

    assert info.runtimes['main'].runtime_id == 'nexus-abc123'
