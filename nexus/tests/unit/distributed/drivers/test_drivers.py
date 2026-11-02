from __future__ import annotations

import asyncio
import hashlib

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from nexus.core import TaskRunInput, TaskRunOutput
from nexus.distributed.config import DistributedConfig
from nexus.distributed.execution.executors.base import ExecutionHandle
from nexus.distributed.execution.trace_backends import RunTraceBackendRegistry, create_run_trace_backend_registry
from nexus.runner.live.trace import LiveRunTraceView
from nexus.runner.single import RunHookFailure, RunOutput
from nexus.runner.trace import RunTraceInMemoryJournal, RunTraceProcessor, RunTraceSession
from nexus.trace.controller import SpanController


def make_trace_registry(root: Path) -> RunTraceBackendRegistry:
    return create_run_trace_backend_registry(DistributedConfig(run_trace_artifact_root=str(root)))


def empty_trace_registry() -> RunTraceBackendRegistry:
    return RunTraceBackendRegistry(default_sink_names=())


@dataclass
class CallbackRecorder:
    running: asyncio.Event = field(default_factory=asyncio.Event)
    completed: asyncio.Event = field(default_factory=asyncio.Event)
    killed: asyncio.Event = field(default_factory=asyncio.Event)
    failed: asyncio.Event = field(default_factory=asyncio.Event)
    waiting: asyncio.Event = field(default_factory=asyncio.Event)
    running_payload: dict[str, Any] | None = None
    waiting_payload: dict[str, Any] | None = None
    runtime_info_updates: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    failure_result: dict[str, Any] | None = None
    completion_runtime_info: dict[str, Any] | None = None
    failure_runtime_info: dict[str, Any] | None = None

    async def mark_running(self, *, endpoint_url=None, endpoint_ready=False, runtime_info=None):
        self.running_payload = {
            'endpoint_url': endpoint_url,
            'endpoint_ready': endpoint_ready,
            'runtime_info': runtime_info or {},
        }
        self.running.set()

    async def mark_waiting_input(self, payload):
        self.waiting_payload = payload
        self.waiting.set()

    async def update_runtime_info(self, runtime_info):
        self.runtime_info_updates.append(runtime_info)

    async def complete(self, result, *, runtime_info=None):
        self.result = result
        self.completion_runtime_info = runtime_info
        self.completed.set()

    async def fail(self, error, *, result=None, runtime_info=None):
        self.error = error
        self.failure_result = result
        self.failure_runtime_info = runtime_info
        self.failed.set()

    async def mark_killed(self):
        self.killed.set()


class SessionBackedLiveRun:
    scenario: ClassVar[list[tuple[str, dict[str, Any]]]] = []

    def __init__(self, request, *, trace_backend_sinks=None, trace_metadata=None):  # noqa: ANN001
        self.run_id = request.run_id
        self.status = 'starting'
        self.current_turn_id = None
        self.trace_records = RunTraceInMemoryJournal()
        self.events = LiveRunTraceView()
        controller = SpanController()
        processor = RunTraceProcessor()
        controller.add_processor(processor)
        self._trace_session = RunTraceSession(
            run_id=self.run_id,
            run_kind='live',
            processor=processor,
            span_controller=controller,
            backend_sinks=dict(trace_backend_sinks or {}),
            internal_sinks=[self.trace_records, self.events],
            metadata=trace_metadata,
        )
        self.submitted: list[str] = []

    @property
    def run_span_id(self) -> str:
        return self._trace_session.run_span_id

    async def start(self, *, wait_until_ready=True):  # noqa: ARG002
        await self._trace_session.start()
        for event_type, data in self.scenario:
            if event_type == 'run.status':
                self.status = str(data.get('status'))
            await self._trace_session.emit(event_type, run_id=self.run_id, **data)
        if self.status in {'closed', 'failed', 'killed'}:
            await self._trace_session.close()

    async def kill(self):
        if self.status not in {'closed', 'failed', 'killed'}:
            self.status = 'killed'
            await self._trace_session.emit('run.killed', run_id=self.run_id)
            await self._trace_session.emit('run.status', run_id=self.run_id, status='killed')
            await self._trace_session.close()

    async def wait_trace_closed(self):
        await self._trace_session.close()

    async def submit_turn(self, request):  # noqa: ANN001
        self.submitted.append(request.text)

    def get_runtime(self, name):  # noqa: ANN001
        return SimpleNamespace(base_url='http://runtime')

    def get_info(self):
        return SimpleNamespace(
            status=self.status, events_cursor=self.events.latest_id, current_turn_id=None, runtimes={}
        )

    def trace_metadata(self) -> dict[str, Any]:
        return self._trace_session.artifact_metadata()


def test_nexus_single_driver_does_not_inject_gongfeng_recovery_keys() -> None:
    from nexus.distributed.execution.drivers.nexus_single import NexusSingleDriver

    driver = NexusSingleDriver(
        execution_id='exec-1',
        assignment_id='asn-3',
        attempt_id='exec-1-attempt-3',
        trace_registry=empty_trace_registry(),
    )

    run_input = driver._build_run_input({
        'task_name': 'identity',
        'runtime_provider_name': 'gongfeng',
    })

    assert 'runtime_recovery_keys' not in run_input.runtime_provider_config


def test_single_driver_forces_execution_id_as_run_id_across_infrastructure_attempts() -> None:
    from nexus.distributed.execution.drivers.nexus_single import NexusSingleDriver

    first = NexusSingleDriver(
        execution_id='exec-stable',
        assignment_id='asn-1',
        attempt_id='exec-stable-attempt-1',
        trace_registry=empty_trace_registry(),
    )
    retry = NexusSingleDriver(
        execution_id='exec-stable',
        assignment_id='asn-2',
        attempt_id='exec-stable-attempt-2',
        trace_registry=empty_trace_registry(),
    )

    assert first._build_run_input({'run_id': 'caller-controlled', 'task_name': 'identity'}).run_id == 'exec-stable'
    assert retry._build_run_input({'run_id': 'different-caller-value', 'task_name': 'identity'}).run_id == 'exec-stable'


def test_live_driver_forces_execution_id_as_run_id_across_infrastructure_attempts() -> None:
    from nexus.distributed.execution.drivers.nexus_live import NexusLiveDriver

    first = NexusLiveDriver(
        execution_id='exec-live-stable',
        assignment_id='asn-1',
        attempt_id='exec-live-stable-attempt-1',
        internal_base_url='http://manager',
        trace_registry=empty_trace_registry(),
    )
    retry = NexusLiveDriver(
        execution_id='exec-live-stable',
        assignment_id='asn-2',
        attempt_id='exec-live-stable-attempt-2',
        internal_base_url='http://manager',
        trace_registry=empty_trace_registry(),
    )

    assert first._build_request({'run_id': 'caller-controlled', 'task_name': 'identity'}, {}).run_id == (
        'exec-live-stable'
    )
    assert retry._build_request({'run_id': 'different-caller-value', 'task_name': 'identity'}, {}).run_id == (
        'exec-live-stable'
    )


@pytest.mark.asyncio
async def test_single_driver_externalizes_complete_output_and_parents_task(monkeypatch, tmp_path) -> None:
    from nexus.distributed.execution.drivers import nexus_single
    from nexus.distributed.execution.drivers.nexus_single import NexusSingleDriver
    from nexus.trace import get_current_span

    seen: dict[str, Any] = {}

    class Runtime:
        async def drain_trace(self) -> None:
            seen['trace_drains'] = seen.get('trace_drains', 0) + 1

    runtime = Runtime()
    context = SimpleNamespace(
        runtime_provider=SimpleNamespace(get_active_runtimes=lambda: {'main': runtime}),
    )

    async def fake_run_single(run_input, hooks=None):  # noqa: ANN001
        task_parent = get_current_span()
        seen['parent_name'] = task_parent.name if task_parent else None
        output = RunOutput(success=True, task_output={'ok': True}, processor_results={'trace': {'tokens': 1}})
        await hooks.on_run_output(context, output)
        await hooks.on_context_closing(context)
        return output

    monkeypatch.setattr(nexus_single, 'run_single', fake_run_single)
    callbacks = CallbackRecorder()
    driver = NexusSingleDriver(
        execution_id='exec-1',
        assignment_id='asn-1',
        attempt_id='attempt-1',
        trace_registry=make_trace_registry(tmp_path),
    )
    await driver.start(
        ExecutionHandle(execution_id='exec-1', assignment_id='asn-1'),
        {'task_name': 'identity'},
        {},
        callbacks,
    )

    assert callbacks.completed.is_set()
    assert callbacks.result['kind'] == 'nexus_run_trace_result'
    assert callbacks.result['metadata']['version'] == 4
    assert callbacks.result['metadata']['run_id'] == 'exec-1'
    assert callbacks.runtime_info_updates == []
    assert callbacks.completion_runtime_info['_artifacts']['single'] == callbacks.result['metadata']
    assert seen['parent_name'] == 'run.single'
    assert seen['trace_drains'] == 2
    assert await driver.trace_registry.read_result(callbacks.result['metadata']) == {
        'success': True,
        'error': None,
        'task_output': {'ok': True},
        'agent_output': None,
        'processor_results': {'trace': {'tokens': 1}},
        'hook_results': {},
        'hook_errors': {},
        'cache_stats': None,
        'workflow_states': {},
        'hook_results': {},
        'hook_errors': {},
    }


@pytest.mark.asyncio
async def test_single_driver_inlines_result_without_artifact_scope(monkeypatch, tmp_path) -> None:
    from nexus.distributed.execution.drivers import nexus_single
    from nexus.distributed.execution.drivers.nexus_single import NexusSingleDriver

    output = RunOutput(success=True, task_output={'ok': True})

    async def fake_run_single(run_input, hooks=None):  # noqa: ANN001
        await hooks.on_run_output(None, output)
        return output

    monkeypatch.setattr(nexus_single, 'run_single', fake_run_single)
    callbacks = CallbackRecorder()
    driver = NexusSingleDriver(
        execution_id='legacy-single',
        assignment_id='legacy-asn',
        trace_registry=make_trace_registry(tmp_path),
        artifact_scope_available=False,
    )

    await driver.start(
        ExecutionHandle(execution_id='legacy-single', assignment_id='legacy-asn'),
        {'task_name': 'identity'},
        {},
        callbacks,
    )

    assert callbacks.result == output.model_dump(mode='json')


@pytest.mark.asyncio
async def test_batch_child_single_driver_propagates_parent_to_trace_context(monkeypatch, tmp_path) -> None:
    from nexus.distributed.execution.drivers import nexus_single
    from nexus.distributed.execution.drivers.nexus_single import NexusSingleDriver

    contexts = []
    registry = make_trace_registry(tmp_path)
    original_resolve = registry.resolve

    def resolve(**kwargs):  # noqa: ANN003
        resolved = original_resolve(**kwargs)
        sink = next(iter(resolved.backend_sinks.values()))
        original_start = sink.start

        async def start(context):  # noqa: ANN001
            contexts.append(context)
            await original_start(context)

        sink.start = start
        return resolved

    async def fake_run_single(run_input, hooks=None):  # noqa: ANN001, ARG001
        return RunOutput(success=True, task_output={'ok': True})

    monkeypatch.setattr(registry, 'resolve', resolve)
    monkeypatch.setattr(nexus_single, 'run_single', fake_run_single)
    callbacks = CallbackRecorder()
    driver = NexusSingleDriver(
        execution_id='exe_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        assignment_id='asn-child',
        owner_principal_id='principal-owner-a',
        parent_execution_id='exe_0123456789abcdef0123456789abcdef',
        trace_registry=registry,
    )

    await driver.start(
        ExecutionHandle(execution_id='exe_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', assignment_id='asn-child'),
        {
            'task_name': 'identity',
            'batch': {
                'parent_execution_id': 'exe_0123456789abcdef0123456789abcdef',
                'item_id': 'item-1',
                'attempt_no': 0,
            },
        },
        {},
        callbacks,
    )

    assert contexts[0].metadata == {
        'execution_id': 'exe_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        'assignment_id': 'asn-child',
        'attempt_id': 'asn-child',
        'storage_scope': 'principal_v1',
        'principal_scope_id': '3190e1c7731fff7995861c3221cec9a6',
        'parent_execution_id': 'exe_0123456789abcdef0123456789abcdef',
    }


@pytest.mark.asyncio
async def test_regular_single_uses_principal_trace_scope(monkeypatch, tmp_path) -> None:
    from nexus.distributed.execution.drivers import nexus_single
    from nexus.distributed.execution.drivers.nexus_single import NexusSingleDriver

    contexts = []
    registry = make_trace_registry(tmp_path)
    original_resolve = registry.resolve

    def resolve(**kwargs):  # noqa: ANN003
        resolved = original_resolve(**kwargs)
        sink = next(iter(resolved.backend_sinks.values()))
        original_start = sink.start

        async def start(context):  # noqa: ANN001
            contexts.append(context)
            await original_start(context)

        sink.start = start
        return resolved

    async def fake_run_single(run_input, hooks=None):  # noqa: ANN001, ARG001
        return RunOutput(success=True, task_output={'ok': True})

    monkeypatch.setattr(registry, 'resolve', resolve)
    monkeypatch.setattr(nexus_single, 'run_single', fake_run_single)
    callbacks = CallbackRecorder()
    driver = NexusSingleDriver(
        execution_id='exe_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
        assignment_id='asn-single',
        owner_principal_id='principal-owner-a',
        trace_registry=registry,
    )

    await driver.start(
        ExecutionHandle(execution_id=driver.execution_id, assignment_id=driver.assignment_id),
        {'task_name': 'identity'},
        {},
        callbacks,
    )

    assert contexts[0].metadata == {
        'execution_id': driver.execution_id,
        'assignment_id': driver.assignment_id,
        'attempt_id': driver.attempt_id,
        'storage_scope': 'principal_v1',
        'principal_scope_id': '3190e1c7731fff7995861c3221cec9a6',
    }


@pytest.mark.asyncio
async def test_batch_child_rejects_parent_mismatch_before_running(monkeypatch, tmp_path) -> None:
    from nexus.distributed.execution.drivers import nexus_single
    from nexus.distributed.execution.drivers.nexus_single import NexusSingleDriver

    called = False

    async def fake_run_single(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(nexus_single, 'run_single', fake_run_single)
    callbacks = CallbackRecorder()
    driver = NexusSingleDriver(
        execution_id='exe_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        assignment_id='asn-child',
        owner_principal_id='principal-owner-a',
        parent_execution_id='exe_0123456789abcdef0123456789abcdef',
        trace_registry=make_trace_registry(tmp_path),
    )

    await driver.start(
        ExecutionHandle(execution_id=driver.execution_id, assignment_id=driver.assignment_id),
        {
            'task_name': 'identity',
            'batch': {'parent_execution_id': 'exe_fedcba9876543210fedcba9876543210'},
        },
        {},
        callbacks,
    )

    assert callbacks.failed.is_set()
    assert callbacks.error is not None
    assert callbacks.error['type'] == 'InvalidExecutionSpec'
    assert 'does not match persisted parent_execution_id' in callbacks.error['message']
    assert called is False


def test_regular_single_rejects_reserved_batch_child_metadata(tmp_path) -> None:
    from nexus.distributed.execution.drivers.nexus_single import NexusSingleDriver

    driver = NexusSingleDriver(
        execution_id='exe_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        assignment_id='asn-single',
        trace_registry=make_trace_registry(tmp_path),
    )

    with pytest.raises(ValueError, match='requires a persisted parent_execution_id'):
        driver._trace_context_metadata({'batch': {'parent_execution_id': 'exe_0123456789abcdef0123456789abcdef'}})


@pytest.mark.asyncio
async def test_single_driver_rejects_empty_sink_selection_before_running(monkeypatch, tmp_path) -> None:
    from nexus.distributed.execution.drivers import nexus_single
    from nexus.distributed.execution.drivers.nexus_single import NexusSingleDriver

    called = False

    async def fake_run_single(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal called
        called = True

    monkeypatch.setattr(nexus_single, 'run_single', fake_run_single)
    callbacks = CallbackRecorder()
    driver = NexusSingleDriver(execution_id='exec', assignment_id='asn', trace_registry=make_trace_registry(tmp_path))
    await driver.start(
        ExecutionHandle(execution_id='exec', assignment_id='asn'),
        {'task_name': 'identity'},
        {'run_trace_backends': {'sinks': []}},
        callbacks,
    )

    assert callbacks.failed.is_set()
    assert callbacks.error['type'] == 'InvalidDriverConfig'
    assert called is False


@pytest.mark.asyncio
async def test_single_driver_reports_failed_run_output(monkeypatch, tmp_path) -> None:
    from nexus.distributed.execution.drivers import nexus_single
    from nexus.distributed.execution.drivers.nexus_single import NexusSingleDriver

    async def fake_run_single(run_input, hooks=None):  # noqa: ANN001
        return RunOutput(success=False, error={'type': 'TaskFailed', 'message': 'nope'})

    monkeypatch.setattr(nexus_single, 'run_single', fake_run_single)
    callbacks = CallbackRecorder()
    driver = NexusSingleDriver(execution_id='exec', assignment_id='asn', trace_registry=make_trace_registry(tmp_path))
    await driver.start(
        ExecutionHandle(execution_id='exec', assignment_id='asn'),
        {'task_name': 'identity'},
        {},
        callbacks,
    )

    assert callbacks.failed.is_set()
    assert callbacks.error == {'type': 'TaskFailed', 'message': 'nope', 'traceback': None}
    assert callbacks.failure_result['kind'] == 'nexus_run_trace_result'
    assert await driver.trace_registry.read_result(callbacks.failure_result['metadata']) == {
        'success': False,
        'error': {'type': 'TaskFailed', 'message': 'nope', 'traceback': None},
        'task_output': None,
        'agent_output': None,
        'processor_results': {},
        'hook_results': {},
        'hook_errors': {},
        'cache_stats': None,
        'workflow_states': {},
        'hook_results': {},
        'hook_errors': {},
    }
    assert callbacks.runtime_info_updates == []
    assert callbacks.failure_runtime_info['_artifacts']['single'] == callbacks.failure_result['metadata']


@pytest.mark.asyncio
async def test_single_driver_fails_without_publishing_trace_when_drain_hook_failed(monkeypatch, tmp_path) -> None:
    from nexus.distributed.execution.drivers import nexus_single
    from nexus.distributed.execution.drivers.nexus_single import NexusSingleDriver

    async def fake_run_single(run_input, hooks=None):  # noqa: ANN001
        hooks.hook_failures.append(
            RunHookFailure(stage='context_closing', error=RuntimeError('final trace drain failed'))
        )
        return RunOutput(success=True, task_output={'ok': True})

    monkeypatch.setattr(nexus_single, 'run_single', fake_run_single)
    callbacks = CallbackRecorder()
    driver = NexusSingleDriver(execution_id='exec', assignment_id='asn', trace_registry=make_trace_registry(tmp_path))
    await driver.start(
        ExecutionHandle(execution_id='exec', assignment_id='asn'),
        {'task_name': 'identity'},
        {},
        callbacks,
    )

    assert callbacks.failed.is_set()
    assert callbacks.completed.is_set() is False
    assert callbacks.error['type'] == 'RunHookError'
    assert 'context_closing: final trace drain failed' in callbacks.error['message']
    assert callbacks.failure_result is None
    assert callbacks.runtime_info_updates == []


@pytest.mark.asyncio
@pytest.mark.parametrize('driver_kind', ['single', 'live'])
@pytest.mark.parametrize('failure', ['drain', 'cleanup', 'sink_write', 'sink_close', 'upload'])
async def test_result_artifact_survives_trace_failure_and_upload_failure_remains_fatal(
    monkeypatch, driver_kind, failure
):
    from nexus.distributed.artifacts.run_result import ArtifactTraceSink, ArtifactTraceSource
    from nexus.distributed.execution.drivers.nexus_live import NexusLiveDriver
    from nexus.distributed.execution.drivers.nexus_single import NexusSingleDriver
    from nexus.distributed.execution.trace_backends import RunTraceBackendRegistration
    from nexus.distributed.trace.bindings import validate_source_binding_update
    from nexus.runner.single import _invoke_post_run_hook
    from nexus.runner.trace import RunTraceSink

    class Client:
        def __init__(self):
            self.content = None
            self.publishes = 0

        async def publish_run_result(self, *, content, **kwargs):
            self.publishes += 1
            if failure == 'upload':
                raise PermissionError('result upload denied')
            self.content = content
            return {
                'artifact_id': 'art-result',
                'kind': 'run_result',
                'sha256': hashlib.sha256(content).hexdigest(),
                'size_bytes': len(content),
            }

        async def read_run_result(self, **kwargs):
            return self.content

    client = Client()
    trace_records = []

    class TraceSink(RunTraceSink):
        async def start(self, context):
            pass

        async def on_event(self, context, record):
            if failure == 'sink_write' and any(
                (e.get('attributes') or {}).get('event_type') == 'run.output'
                for e in record.event_data.get('events') or []
            ):
                raise RuntimeError('trace DB rejected output')
            trace_records.append(record)

        async def close(self, context):
            if failure == 'sink_close':
                assert client.content is not None, 'Result publication must precede raw trace close'
                raise RuntimeError('trace DB close timeout')

    class Runtime:
        async def drain_trace(self):
            if failure == 'drain':
                raise RuntimeError('runtime trace ACK timeout')

    runtime = Runtime()
    context = SimpleNamespace(runtime_provider=SimpleNamespace(get_active_runtimes=lambda: {'main': runtime}))
    output = RunOutput(
        success=True, task_output={'archive': 'project.tar.gz'}, processor_results={'chat': {'turns': 9}}
    )

    async def fake_run_single(run_input, hooks=None):
        await _invoke_post_run_hook(hooks, 'run_output', hooks.on_run_output, context, output)
        await _invoke_post_run_hook(hooks, 'context_closing', hooks.on_context_closing, context)
        if failure == 'cleanup':
            raise RuntimeError('runtime cleanup failed')
        return output

    monkeypatch.setattr('nexus.distributed.execution.drivers.nexus_single.run_single', fake_run_single)
    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)
    registry = RunTraceBackendRegistry(default_sink_names=['tdsql', 'artifact'])

    class MissingTraceSource(ArtifactTraceSource):
        async def read_result(self, metadata):
            raise FileNotFoundError('trace has no output')

        async def read_events_after(self, metadata, cursor, *, limit=None):
            records = [record for record in trace_records if record.record_id > cursor]
            return records if limit is None else records[:limit]

    source = ArtifactTraceSource(client=client)
    registry.register(
        RunTraceBackendRegistration('tdsql', lambda request: TraceSink(), MissingTraceSource(client=client))
    )
    registry.register(
        RunTraceBackendRegistration(
            'artifact',
            lambda request: ArtifactTraceSink(
                client=client,
                execution_id=request.execution_id,
                assignment_id=request.assignment_id,
                attempt_id=request.attempt_id,
                compression='zstd',
            ),
            source,
        )
    )

    class ValidatingCallbacks(CallbackRecorder):
        async def fail(self, error, *, result=None, runtime_info=None):
            if driver_kind == 'single':
                validate_source_binding_update(
                    self.running_payload['runtime_info'],
                    runtime_info,
                    execution_id=driver.execution_id,
                    assignment_id=driver.assignment_id,
                    attempt_id=driver.attempt_id,
                    owner_principal_id=driver.owner_principal_id,
                )
            await super().fail(error, result=result, runtime_info=runtime_info)

    callbacks = ValidatingCallbacks()
    kwargs = {'execution_id': 'exec', 'assignment_id': 'asn', 'trace_registry': registry}
    driver = (
        NexusSingleDriver(**kwargs)
        if driver_kind == 'single'
        else NexusLiveDriver(**kwargs, internal_base_url='http://manager')
    )
    await driver.start(
        ExecutionHandle(execution_id='exec', assignment_id='asn'),
        {'task_name': 'identity', 'runner': {'trace_drain_error_policy': 'raise'}},
        {},
        callbacks,
    )

    assert callbacks.failed.is_set()
    assert not callbacks.completed.is_set()
    if failure == 'upload':
        assert callbacks.error['type'] == 'ResultPersistenceFailed'
        assert callbacks.failure_result is None
        assert client.content is None
    else:
        assert client.publishes == 1
        assert callbacks.failure_result is not None
        assert await registry.read_result(callbacks.failure_result['metadata']) == output.model_dump(mode='json')
        if failure in {'drain', 'cleanup'}:
            assert {entry['name'] for entry in callbacks.failure_result['metadata']['sinks']} == {'tdsql', 'artifact'}
            if driver_kind == 'live':
                assert await registry.read_live_events_after(callbacks.failure_result['metadata'], 0)
        elif driver_kind == 'single':
            assert callbacks.failure_runtime_info is None


@pytest.mark.asyncio
@pytest.mark.parametrize('policy', [None, 'raise', 'warn'])
@pytest.mark.parametrize('failure_stage', ['run_output', 'context_closing'])
@pytest.mark.parametrize('task_success', [True, False])
async def test_single_driver_trace_drain_policy_preserves_task_result(
    monkeypatch, tmp_path, caplog, policy, failure_stage, task_success
) -> None:
    from socketio.exceptions import TimeoutError as SocketIOTimeoutError

    from nexus.distributed.execution.drivers import nexus_single
    from nexus.distributed.execution.drivers.nexus_single import NexusSingleDriver
    from nexus.runner.single import _invoke_post_run_hook

    current_stage = None
    drains = []

    class Runtime:
        def __init__(self, name):
            self.name = name

        async def drain_trace(self):
            drains.append((current_stage, self.name))
            if self.name == 'main' and current_stage == failure_stage:
                raise SocketIOTimeoutError()

    failing_runtime = Runtime('main')
    runtimes = {'main': failing_runtime, 'alias': failing_runtime, 'eval': Runtime('eval')}
    context = SimpleNamespace(runtime_provider=SimpleNamespace(get_active_runtimes=lambda: runtimes))
    output = RunOutput(
        success=task_success,
        task_output={'archive': 'project.tar.gz'},
        error=None if task_success else {'type': 'TaskFailed', 'message': 'business failure'},
    )

    async def fake_run_single(run_input, hooks=None):
        nonlocal current_stage
        current_stage = 'run_output'
        await _invoke_post_run_hook(hooks, current_stage, hooks.on_run_output, context, output)
        current_stage = 'context_closing'
        await _invoke_post_run_hook(hooks, current_stage, hooks.on_context_closing, context)
        return output

    monkeypatch.setattr(nexus_single, 'run_single', fake_run_single)
    callbacks = CallbackRecorder()
    driver = NexusSingleDriver(execution_id='exec', assignment_id='asn', trace_registry=make_trace_registry(tmp_path))
    spec = {'task_name': 'identity'}
    if policy is not None:
        spec['runner'] = {'trace_drain_error_policy': policy}
    await driver.start(ExecutionHandle(execution_id='exec', assignment_id='asn'), spec, {}, callbacks)

    if policy == 'raise':
        assert callbacks.failed.is_set()
        assert not callbacks.completed.is_set()
        assert callbacks.error['type'] == 'RunHookError'
        return

    assert callbacks.completed.is_set() == task_success
    assert callbacks.failed.is_set() != task_success
    result = callbacks.result if task_success else callbacks.failure_result
    assert result['kind'] == 'nexus_run_trace_result'
    assert await driver.trace_registry.read_result(result['metadata']) == output.model_dump(mode='json')
    if not task_success:
        assert callbacks.error['type'] == 'TaskFailed'
    assert drains == [
        ('run_output', 'main'),
        ('run_output', 'eval'),
        ('context_closing', 'main'),
        ('context_closing', 'eval'),
    ]
    warnings = [record for record in caplog.records if record.name == 'nexus.runtime.trace_drain']
    assert len(warnings) == 1
    assert warnings[0].levelname == 'WARNING'
    assert warnings[0].exc_info[0] is SocketIOTimeoutError
    assert 'run_id=exec runtime=main error_type=TimeoutError' in warnings[0].getMessage()
    assert 'potentially incomplete trace' in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_single_driver_preserves_output_pointer_when_cleanup_fails(monkeypatch, tmp_path) -> None:
    from nexus.distributed.execution.drivers import nexus_single
    from nexus.distributed.execution.drivers.nexus_single import NexusSingleDriver

    class Runtime:
        async def drain_trace(self) -> None:
            return None

    context = SimpleNamespace(
        runtime_provider=SimpleNamespace(get_active_runtimes=lambda: {'main': Runtime()}),
    )
    output = RunOutput(success=True, task_output={'ok': True})

    async def fake_run_single(run_input, hooks=None):  # noqa: ANN001
        await hooks.on_run_output(context, output)
        raise RuntimeError('cleanup failed')

    monkeypatch.setattr(nexus_single, 'run_single', fake_run_single)
    callbacks = CallbackRecorder()
    driver = NexusSingleDriver(execution_id='exec', assignment_id='asn', trace_registry=make_trace_registry(tmp_path))
    await driver.start(
        ExecutionHandle(execution_id='exec', assignment_id='asn'),
        {'task_name': 'identity'},
        {},
        callbacks,
    )

    assert callbacks.failed.is_set()
    assert callbacks.error['type'] == 'RuntimeError'
    assert callbacks.failure_result['kind'] == 'nexus_run_trace_result'
    assert await driver.trace_registry.read_result(callbacks.failure_result['metadata']) == {
        'success': True,
        'error': None,
        'task_output': {'ok': True},
        'agent_output': None,
        'processor_results': {},
        'hook_results': {},
        'hook_errors': {},
        'cache_stats': None,
        'workflow_states': {},
        'hook_results': {},
        'hook_errors': {},
    }


@pytest.mark.asyncio
async def test_single_driver_cancellation_is_not_replaced_by_trace_metadata_failure(monkeypatch, tmp_path) -> None:
    from nexus.distributed.execution.drivers import nexus_single
    from nexus.distributed.execution.drivers.nexus_single import NexusSingleDriver

    async def blocked_run_single(run_input, hooks=None):  # noqa: ANN001, ARG001
        await asyncio.Event().wait()

    monkeypatch.setattr(nexus_single, 'run_single', blocked_run_single)
    callbacks = CallbackRecorder()
    metadata_attempts = 0

    async def fail_metadata_update(runtime_info):  # noqa: ANN001, ARG001
        nonlocal metadata_attempts
        metadata_attempts += 1
        raise OSError('metadata unavailable')

    monkeypatch.setattr(callbacks, 'update_runtime_info', fail_metadata_update)
    driver = NexusSingleDriver(execution_id='exec', assignment_id='asn', trace_registry=make_trace_registry(tmp_path))
    handle = ExecutionHandle(execution_id='exec', assignment_id='asn')
    start_task = asyncio.create_task(driver.start(handle, {'task_name': 'identity'}, {}, callbacks))
    await callbacks.running.wait()

    start_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await start_task
    assert metadata_attempts == 1


@pytest.mark.asyncio
async def test_single_driver_does_not_publish_terminal_trace_metadata_as_a_separate_update(
    monkeypatch, tmp_path
) -> None:
    from nexus.distributed.execution.drivers import nexus_single
    from nexus.distributed.execution.drivers.nexus_single import NexusSingleDriver

    async def fake_run_single(run_input, hooks=None):  # noqa: ANN001, ARG001
        return RunOutput(success=True, task_output={'ok': True})

    monkeypatch.setattr(nexus_single, 'run_single', fake_run_single)
    callbacks = CallbackRecorder()
    driver = NexusSingleDriver(execution_id='exec', assignment_id='asn', trace_registry=make_trace_registry(tmp_path))

    await driver.start(
        ExecutionHandle(execution_id='exec', assignment_id='asn'),
        {'task_name': 'identity'},
        {},
        callbacks,
    )

    assert callbacks.completed.is_set()
    assert callbacks.runtime_info_updates == []
    assert callbacks.completion_runtime_info['_artifacts']['single'] == callbacks.result['metadata']


@pytest.mark.asyncio
async def test_single_driver_maps_run_result_publish_failure(monkeypatch) -> None:
    from nexus.distributed.artifacts.run_result import ResultPersistenceError
    from nexus.distributed.execution.drivers import nexus_single
    from nexus.distributed.execution.drivers.nexus_single import NexusSingleDriver
    from nexus.distributed.execution.trace_backends import RunTraceBackendRegistration
    from nexus.runner.trace import RunTraceSink, RunTraceSource

    class FailingSink(RunTraceSink):
        async def start(self, context) -> None:  # noqa: ANN001
            return None

        async def on_event(self, context, event) -> None:  # noqa: ANN001
            return None

        async def close(self, context) -> None:  # noqa: ANN001
            raise ResultPersistenceError('gateway rejected result')

    class EmptySource(RunTraceSource):
        async def read_events_after(self, metadata, cursor, *, limit=None):  # noqa: ANN001, ARG002
            return []

        async def read_result(self, metadata):  # noqa: ANN001, ARG002
            raise FileNotFoundError

    async def fake_run_single(run_input, hooks=None):  # noqa: ANN001, ARG001
        return RunOutput(success=True, task_output={'ok': True})

    registry = RunTraceBackendRegistry(default_sink_names=['artifact'])
    registry.register(RunTraceBackendRegistration('artifact', lambda request: FailingSink(), EmptySource()))
    monkeypatch.setattr(nexus_single, 'run_single', fake_run_single)
    callbacks = CallbackRecorder()
    driver = NexusSingleDriver(execution_id='exec', assignment_id='asn', trace_registry=registry)

    await driver.start(
        ExecutionHandle(execution_id='exec', assignment_id='asn'),
        {'task_name': 'identity'},
        {},
        callbacks,
    )

    assert callbacks.completed.is_set() is False
    assert callbacks.error == {'type': 'ResultPersistenceFailed', 'message': 'gateway rejected result'}


@pytest.mark.asyncio
async def test_live_driver_consumes_raw_run_events_and_returns_trace_pointer(monkeypatch, tmp_path) -> None:
    from nexus.distributed.execution.drivers import nexus_live
    from nexus.distributed.execution.drivers.nexus_live import NexusLiveDriver

    SessionBackedLiveRun.scenario = [
        ('runtime.ready', {'runtimes': {'main': {'base_url': 'http://rt'}}}),
        ('input.required', {'request': {'hint': 'choose'}}),
        ('run.output', {'output': {'success': True, 'value': 'done'}}),
        ('run.status', {'status': 'closed'}),
    ]
    monkeypatch.setattr(nexus_live, 'LiveRun', SessionBackedLiveRun)
    callbacks = CallbackRecorder()
    driver = NexusLiveDriver(
        execution_id='exec-live',
        assignment_id='asn-live',
        internal_base_url='http://manager/internal',
        trace_registry=make_trace_registry(tmp_path),
    )
    await driver.start(
        ExecutionHandle(execution_id='exec-live', assignment_id='asn-live'),
        {'task_name': 'identity'},
        {},
        callbacks,
    )

    assert callbacks.running.is_set()
    assert callbacks.waiting_payload == {'run_id': 'exec-live', 'request': {'hint': 'choose'}}
    assert callbacks.result['kind'] == 'nexus_live_result'
    envelope = callbacks.result['metadata']
    assert envelope['version'] == 4
    assert envelope['run_id'] == 'exec-live'
    assert await driver.trace_registry.read_result(envelope) == {'success': True, 'value': 'done'}
    assert [event.type for event in await driver.trace_registry.read_live_events_after(envelope, 0)] == [
        'runtime.ready',
        'input.required',
        'run.output',
        'run.status',
    ]
    assert callbacks.runtime_info_updates[-1]['_artifacts']['live'] == envelope


@pytest.mark.asyncio
async def test_live_driver_uses_principal_trace_scope(monkeypatch, tmp_path) -> None:
    from nexus.distributed.execution.drivers import nexus_live
    from nexus.distributed.execution.drivers.nexus_live import NexusLiveDriver

    contexts = []
    registry = make_trace_registry(tmp_path)
    original_resolve = registry.resolve

    def resolve(**kwargs):  # noqa: ANN003
        resolved = original_resolve(**kwargs)
        sink = next(iter(resolved.backend_sinks.values()))
        original_start = sink.start

        async def start(context):  # noqa: ANN001
            contexts.append(context)
            await original_start(context)

        sink.start = start
        return resolved

    SessionBackedLiveRun.scenario = [
        ('run.output', {'output': {'success': True, 'value': 'done'}}),
        ('run.status', {'status': 'closed'}),
    ]
    monkeypatch.setattr(registry, 'resolve', resolve)
    monkeypatch.setattr(nexus_live, 'LiveRun', SessionBackedLiveRun)
    callbacks = CallbackRecorder()
    driver = NexusLiveDriver(
        execution_id='exe_cccccccccccccccccccccccccccccccc',
        assignment_id='asn-live',
        owner_principal_id='principal-owner-a',
        internal_base_url='http://manager/internal',
        trace_registry=registry,
    )

    await driver.start(
        ExecutionHandle(execution_id=driver.execution_id, assignment_id=driver.assignment_id),
        {'task_name': 'identity'},
        {},
        callbacks,
    )

    assert contexts[0].metadata == {
        'execution_id': driver.execution_id,
        'assignment_id': driver.assignment_id,
        'attempt_id': driver.attempt_id,
        'storage_scope': 'principal_v1',
        'principal_scope_id': '3190e1c7731fff7995861c3221cec9a6',
    }


@pytest.mark.asyncio
async def test_live_driver_inlines_result_without_artifact_scope(monkeypatch, tmp_path) -> None:
    from nexus.distributed.execution.drivers import nexus_live
    from nexus.distributed.execution.drivers.nexus_live import NexusLiveDriver

    SessionBackedLiveRun.scenario = [
        ('run.output', {'output': {'success': True, 'value': 'done'}}),
        ('run.status', {'status': 'closed'}),
    ]
    monkeypatch.setattr(nexus_live, 'LiveRun', SessionBackedLiveRun)
    callbacks = CallbackRecorder()
    driver = NexusLiveDriver(
        execution_id='legacy-live',
        assignment_id='legacy-asn',
        internal_base_url='http://manager',
        trace_registry=make_trace_registry(tmp_path),
        artifact_scope_available=False,
    )

    await driver.start(
        ExecutionHandle(execution_id='legacy-live', assignment_id='legacy-asn'),
        {'task_name': 'identity'},
        {},
        callbacks,
    )

    assert callbacks.result == {'success': True, 'value': 'done'}


@pytest.mark.asyncio
async def test_live_driver_maps_failed_output_to_failure(monkeypatch, tmp_path) -> None:
    from nexus.distributed.execution.drivers import nexus_live
    from nexus.distributed.execution.drivers.nexus_live import NexusLiveDriver

    SessionBackedLiveRun.scenario = [
        ('run.output', {'output': {'success': False, 'error': {'type': 'TaskFailed', 'message': 'nope'}}}),
        ('run.status', {'status': 'failed'}),
    ]
    monkeypatch.setattr(nexus_live, 'LiveRun', SessionBackedLiveRun)
    callbacks = CallbackRecorder()
    driver = NexusLiveDriver(
        execution_id='exec',
        assignment_id='asn',
        internal_base_url='http://manager',
        trace_registry=make_trace_registry(tmp_path),
    )
    await driver.start(
        ExecutionHandle(execution_id='exec', assignment_id='asn'),
        {'task_name': 'identity'},
        {},
        callbacks,
    )

    assert callbacks.failed.is_set()
    assert callbacks.error == {'type': 'TaskFailed', 'message': 'nope'}
    assert callbacks.failure_result['kind'] == 'nexus_live_result'
    assert await driver.trace_registry.read_result(callbacks.failure_result['metadata']) == {
        'success': False,
        'error': {'type': 'TaskFailed', 'message': 'nope'},
    }


@pytest.mark.asyncio
async def test_live_driver_status_fallback_reports_live_run_error_as_cause(tmp_path) -> None:
    # When the run failed without a terminal error event (e.g. trace pipeline collapsed after a
    # runtime Socket.IO disconnect), the status fallback must surface the live run's recorded
    # cause instead of the generic "live run failed".
    from nexus.distributed.execution.drivers.nexus_live import NexusLiveDriver

    class _FailedLiveRun:
        status = 'failed'
        error = 'RunHookError: context_closing: runtime Socket.IO disconnected before trace drain'

        async def wait_trace_closed(self) -> None:
            return None

        def trace_metadata(self) -> dict[str, Any]:
            return {}

    driver = NexusLiveDriver(
        execution_id='exec',
        assignment_id='asn',
        internal_base_url='http://manager',
        trace_registry=make_trace_registry(tmp_path),
    )
    live_run = _FailedLiveRun()
    driver._live_run = live_run
    callbacks = CallbackRecorder()

    await driver._report_failed_live_run(live_run, callbacks, None)

    assert callbacks.failed.is_set()
    assert callbacks.error['type'] == 'RunFailed'
    assert 'Socket.IO disconnected before trace drain' in callbacks.error['message']


@pytest.mark.asyncio
async def test_live_driver_sink_failure_is_not_replaced_with_inline_success(monkeypatch, tmp_path) -> None:
    import nexus.runner.trace.local as trace_local

    from nexus.distributed.execution.drivers import nexus_live
    from nexus.distributed.execution.drivers.nexus_live import NexusLiveDriver

    monkeypatch.setattr(
        trace_local, '_append_records', lambda path, records: (_ for _ in ()).throw(OSError('disk full'))
    )
    SessionBackedLiveRun.scenario = [
        ('run.output', {'output': {'success': True}}),
        ('run.status', {'status': 'closed'}),
    ]
    monkeypatch.setattr(nexus_live, 'LiveRun', SessionBackedLiveRun)
    callbacks = CallbackRecorder()
    driver = NexusLiveDriver(
        execution_id='exec',
        assignment_id='asn',
        internal_base_url='http://manager',
        trace_registry=make_trace_registry(tmp_path),
    )
    await driver.start(
        ExecutionHandle(execution_id='exec', assignment_id='asn'),
        {'task_name': 'identity'},
        {},
        callbacks,
    )

    assert callbacks.failed.is_set()
    assert callbacks.completed.is_set() is False


@pytest.mark.asyncio
async def test_live_driver_maps_run_result_publish_failure(monkeypatch) -> None:
    from nexus.distributed.artifacts.run_result import ResultPersistenceError
    from nexus.distributed.execution.drivers import nexus_live
    from nexus.distributed.execution.drivers.nexus_live import NexusLiveDriver
    from nexus.distributed.execution.trace_backends import RunTraceBackendRegistration
    from nexus.runner.trace import RunTraceSink, RunTraceSource

    class FailingSink(RunTraceSink):
        async def start(self, context) -> None:  # noqa: ANN001
            return None

        async def on_event(self, context, event) -> None:  # noqa: ANN001
            return None

        async def close(self, context) -> None:  # noqa: ANN001
            raise ResultPersistenceError('gateway rejected result')

    class EmptySource(RunTraceSource):
        async def read_events_after(self, metadata, cursor, *, limit=None):  # noqa: ANN001, ARG002
            return []

        async def read_result(self, metadata):  # noqa: ANN001, ARG002
            raise FileNotFoundError

    registry = RunTraceBackendRegistry(default_sink_names=['artifact'])
    registry.register(RunTraceBackendRegistration('artifact', lambda request: FailingSink(), EmptySource()))
    SessionBackedLiveRun.scenario = [
        ('run.output', {'output': {'success': True}}),
        ('run.status', {'status': 'closed'}),
    ]
    monkeypatch.setattr(nexus_live, 'LiveRun', SessionBackedLiveRun)
    callbacks = CallbackRecorder()
    driver = NexusLiveDriver(
        execution_id='exec',
        assignment_id='asn',
        internal_base_url='http://manager',
        trace_registry=registry,
    )

    await driver.start(
        ExecutionHandle(execution_id='exec', assignment_id='asn'),
        {'task_name': 'identity'},
        {},
        callbacks,
    )

    assert callbacks.completed.is_set() is False
    assert callbacks.error == {'type': 'ResultPersistenceFailed', 'message': 'gateway rejected result'}


@pytest.mark.asyncio
async def test_live_driver_commands_sse_and_runtime_proxy_use_live_view(monkeypatch, tmp_path) -> None:
    from nexus.distributed.execution.drivers import nexus_live
    from nexus.distributed.execution.drivers.nexus_live import NexusLiveDriver

    SessionBackedLiveRun.scenario = [('runtime.ready', {'runtimes': {}})]
    monkeypatch.setattr(nexus_live, 'LiveRun', SessionBackedLiveRun)
    callbacks = CallbackRecorder()
    driver = NexusLiveDriver(
        execution_id='exec',
        assignment_id='asn',
        internal_base_url='http://manager',
        trace_registry=make_trace_registry(tmp_path),
    )
    handle = ExecutionHandle(execution_id='exec', assignment_id='asn')
    start_task = asyncio.create_task(driver.start(handle, {'task_name': 'identity'}, {}, callbacks))
    await callbacks.running.wait()

    await driver.submit_command(handle, 'input.submit', {'text': 'hello'})
    with pytest.raises(RuntimeError, match='does not accept command'):
        await driver.submit_command(handle, 'unknown', {})
    assert driver._live_run.submitted == ['hello']
    assert await driver.get_runtime_base_url('main') == 'http://runtime'
    sse = await driver.collect_sse_events_after(0, timeout=0.1)
    assert b'event: runtime.ready' in sse[0][1]

    await driver.stop(handle)
    await start_task
    assert callbacks.killed.is_set()


@pytest.mark.asyncio
async def test_live_driver_cancellation_is_not_replaced_by_kill_failure(monkeypatch, tmp_path) -> None:
    from nexus.distributed.execution.drivers import nexus_live
    from nexus.distributed.execution.drivers.nexus_live import NexusLiveDriver

    class KillFailingLiveRun(SessionBackedLiveRun):
        async def kill(self) -> None:
            await self._trace_session.close()
            raise OSError('kill failed')

    KillFailingLiveRun.scenario = [('runtime.ready', {'runtimes': {}})]
    monkeypatch.setattr(nexus_live, 'LiveRun', KillFailingLiveRun)
    callbacks = CallbackRecorder()
    driver = NexusLiveDriver(
        execution_id='exec',
        assignment_id='asn',
        internal_base_url='http://manager',
        trace_registry=make_trace_registry(tmp_path),
    )
    handle = ExecutionHandle(execution_id='exec', assignment_id='asn')
    start_task = asyncio.create_task(driver.start(handle, {'task_name': 'identity'}, {}, callbacks))
    await callbacks.running.wait()

    start_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await start_task


@pytest.mark.asyncio
async def test_live_driver_rejects_missing_default_sink(monkeypatch) -> None:
    from nexus.distributed.execution.drivers import nexus_live
    from nexus.distributed.execution.drivers.nexus_live import NexusLiveDriver

    monkeypatch.setattr(nexus_live, 'LiveRun', SessionBackedLiveRun)
    callbacks = CallbackRecorder()
    driver = NexusLiveDriver(
        execution_id='exec',
        assignment_id='asn',
        internal_base_url='http://manager',
        trace_registry=empty_trace_registry(),
    )
    await driver.start(
        ExecutionHandle(execution_id='exec', assignment_id='asn'),
        {'task_name': 'identity'},
        {},
        callbacks,
    )
    assert callbacks.failed.is_set()
    assert callbacks.error['type'] == 'InvalidDriverConfig'


@pytest.mark.parametrize('should_fail', [False, True])
async def test_extra_survives_real_runner_trace_and_cloud_export_across_attempts(tmp_path, should_fail):
    import json

    from nexus.core import task_registry
    from nexus.distributed.execution.drivers.nexus_single import NexusSingleDriver
    from nexus.distributed.export.results import make_jsonl_record
    from nexus.distributed.export.writer import _validate_jsonl_file
    from nexus.runner.single import RunInput

    name = f'test_extra_driver_{should_fail}'

    @task_registry.register(name)
    async def task(task_input: ExtraTaskInput) -> ExtraTaskOutput:
        if should_fail:
            raise RuntimeError('expected task failure')
        return ExtraTaskOutput()

    extra = {'text': 'x' * (2 * 1024**2 - 11)}
    raw = RunInput(task_name=name, extra=extra).model_dump(mode='json')
    for attempt in (1, 2):
        registry = make_trace_registry(tmp_path / str(attempt))
        callbacks = CallbackRecorder()
        driver = NexusSingleDriver(execution_id='extra-run', assignment_id=f'asn-{attempt}', trace_registry=registry)
        await driver.start(ExecutionHandle(execution_id='extra-run', assignment_id=f'asn-{attempt}'), {'run_input': raw}, {}, callbacks)
        pointer = callbacks.failure_result if should_fail else callbacks.result
        assert pointer is not None
        output = await registry.read_result(pointer['metadata'])
        assert output['extra'] == extra
        assert output['success'] is (not should_fail)
        record = make_jsonl_record(run_id='case', distributed_status='failed' if should_fail else 'succeeded', start_time=None, end_time=None, retry_count=attempt - 1, fallback_error=None, output=output, metadata={'tag': 1})
        assert record['extra'] == extra
        assert record['metadata'] == {'tag': 1}
        path = tmp_path / f'result-{attempt}.jsonl'
        path.write_text(json.dumps(record) + '\n')
        _validate_jsonl_file(path, 1, False)


class ExtraTaskInput(TaskRunInput):
    pass


class ExtraTaskOutput(TaskRunOutput):
    pass
