from __future__ import annotations

import asyncio
import json
import sys
import threading

from types import ModuleType

import pytest

from nexus.runner.trace import RunTraceContext, RunTraceSink, RunTraceSource
from nexus.trace import SpanEventRecord
from nexus.trace.events import SpanEndEvent, SpanStartEvent, SpanUpdateEvent


def _record(record_id: int, event: SpanStartEvent | SpanUpdateEvent | SpanEndEvent) -> SpanEventRecord:
    return SpanEventRecord(
        record_id=record_id,
        event_type={'START': 'start', 'UPDATE': 'update', 'END': 'end'}[event.type],
        time_unix_nano=event.time_unix_nano,
        trace_id=getattr(event, 'trace_id', 'trace-run'),
        span_id=event.span_id,
        parent_span_id=getattr(event, 'parent_span_id', ''),
        event_data=event.model_dump(mode='json'),
    )


def _output_record(record_id: int, span_id: str, output: dict) -> SpanEventRecord:
    return SpanEventRecord(
        record_id=record_id,
        event_type='update',
        time_unix_nano=record_id,
        trace_id='trace-1',
        span_id=span_id,
        parent_span_id='',
        event_data={
            'events': [
                {
                    'name': 'run.event',
                    'time_unix_nano': record_id,
                    'attributes': {'event_type': 'run.output', 'event_data': {'output': output}},
                }
            ]
        },
    )


@pytest.mark.asyncio
async def test_local_sink_writes_raw_records_and_source_reads_result(tmp_path) -> None:
    from nexus.runner.trace import RunTraceLocalSink, RunTraceLocalSource

    context = RunTraceContext(run_id='exec-1', run_span_id='run-span-1', run_kind='single')
    sink = RunTraceLocalSink(
        root=tmp_path,
        execution_id='exec-1',
        assignment_id='assign-1',
        attempt_id='attempt-1',
    )
    source = RunTraceLocalSource(root=tmp_path)
    start = SpanStartEvent(
        span_id='run-span-1',
        trace_id='trace-run',
        parent_span_id='',
        name='run.single',
        attributes={'run_id': 'exec-1', 'run_kind': 'single'},
    )
    records = [_record(1, start), _output_record(2, 'run-span-1', {'success': True})]

    await sink.start(context)
    for record in records:
        await sink.on_event(context, record)
    await sink.close(context)

    metadata = sink.metadata()
    path = tmp_path / 'exec-1' / 'attempt-1' / 'assign-1' / 'trace_events.jsonl'
    assert metadata['trace_events_path'] == str(path)
    assert [json.loads(line)['record_id'] for line in path.read_text().splitlines()] == [1, 2]
    assert await source.read_events_after(metadata, 0, limit=1) == records[:1]
    assert await source.read_result(metadata) == {'success': True}


@pytest.mark.asyncio
async def test_local_source_only_reads_output_from_synthetic_run_span(tmp_path) -> None:
    from nexus.runner.trace import RunTraceLocalSink, RunTraceLocalSource

    context = RunTraceContext(run_id='run-1', run_span_id='run-span-1', run_kind='single')
    sink = RunTraceLocalSink(root=tmp_path, execution_id='exec', assignment_id='asn', attempt_id='attempt')
    source = RunTraceLocalSource(root=tmp_path)
    await sink.start(context)
    await sink.on_event(context, _output_record(1, 'run-span-1', {}))
    await sink.on_event(context, _output_record(2, 'ordinary-span', {'wrong': True}))
    await sink.close(context)

    assert await source.read_result(sink.metadata()) == {}


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('execution_id', ''),
        ('execution_id', '.'),
        ('execution_id', '..'),
        ('execution_id', 'a/b'),
        ('attempt_id', 'a\\b'),
        ('assignment_id', 'a\x00b'),
        ('assignment_id', 'a' * 129),
    ],
)
def test_local_sink_rejects_unsafe_path_segments(tmp_path, field, value) -> None:
    from nexus.runner.trace import RunTraceLocalSink

    arguments = {
        'root': tmp_path,
        'execution_id': 'exec-1',
        'assignment_id': 'assign-1',
        'attempt_id': 'attempt-1',
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=field):
        RunTraceLocalSink(**arguments)


@pytest.mark.asyncio
async def test_local_sink_surfaces_writer_failure(monkeypatch, tmp_path) -> None:
    import nexus.runner.trace.local as trace_local

    from nexus.runner.trace import RunTraceLocalSink

    monkeypatch.setattr(
        trace_local, '_append_records', lambda path, records: (_ for _ in ()).throw(OSError('disk full'))
    )
    sink = RunTraceLocalSink(root=tmp_path, execution_id='exec', assignment_id='asn', attempt_id='attempt')
    context = RunTraceContext(run_id='run', run_span_id='span', run_kind='single')
    await sink.start(context)
    await sink.on_event(context, _output_record(1, 'span', {'success': True}))

    with pytest.raises(OSError, match='disk full'):
        await sink.close(context)


@pytest.mark.asyncio
async def test_local_sink_close_does_not_deadlock_when_writer_fails_with_full_queue(monkeypatch, tmp_path) -> None:
    import nexus.runner.trace.local as trace_local

    from nexus.runner.trace import RunTraceLocalSink

    entered = threading.Event()
    release = threading.Event()

    def blocked_failure(path, records):  # noqa: ANN001, ARG001
        entered.set()
        release.wait(timeout=2)
        raise OSError('writer failed')

    monkeypatch.setattr(trace_local, '_append_records', blocked_failure)
    sink = RunTraceLocalSink(
        root=tmp_path,
        execution_id='exec',
        assignment_id='asn',
        attempt_id='attempt',
        batch_size=1,
        queue_maxsize=1,
    )
    context = RunTraceContext(run_id='run', run_span_id='span', run_kind='single')
    await sink.start(context)
    await sink.on_event(context, _output_record(1, 'span', {'first': True}))
    assert await asyncio.to_thread(entered.wait, 1)
    await sink.on_event(context, _output_record(2, 'span', {'second': True}))
    release.set()
    while sink._task is not None and not sink._task.done():
        await asyncio.sleep(0)

    with pytest.raises(OSError, match='writer failed'):
        await asyncio.wait_for(sink.close(context), timeout=1)


@pytest.mark.asyncio
async def test_registry_resolves_multiple_sinks_with_identical_records(tmp_path) -> None:
    from nexus.distributed.config import DistributedConfig
    from nexus.distributed.execution.trace_backends import (
        ResolvedRunTraceBackends,
        RunTraceSinkRequest,
        create_run_trace_backend_registry,
    )
    from nexus.runner.trace import RunTraceLocalSource, RunTraceProcessor, RunTraceSession
    from nexus.trace.controller import SpanController

    class_uri = 'nexus.runner.trace.local::RunTraceLocalSink'
    source_uri = 'nexus.runner.trace.local::RunTraceLocalSource'
    config = DistributedConfig(
        run_trace_sinks={
            'primary': {'sinkClass': class_uri, 'sourceClass': source_uri, 'config': {'root': str(tmp_path / 'a')}},
            'mirror': {'sinkClass': class_uri, 'sourceClass': source_uri, 'config': {'root': str(tmp_path / 'b')}},
        },
        default_run_trace_sinks=['primary', 'mirror'],
    )
    registry = create_run_trace_backend_registry(config)
    backends = registry.resolve(
        run_kind='single',
        driver_config={},
        request=RunTraceSinkRequest(execution_id='exec', assignment_id='asn', attempt_id='attempt'),
    )
    assert isinstance(backends, ResolvedRunTraceBackends)
    controller = SpanController()
    processor = RunTraceProcessor()
    controller.add_processor(processor)
    session = RunTraceSession(
        run_id='exec',
        run_kind='single',
        run_span_id='run-span',
        processor=processor,
        span_controller=controller,
        backend_sinks=backends.backend_sinks,
    )
    await session.start()
    await session.emit('run.output', output={'success': True})
    await session.close()

    envelope = session.artifact_metadata()
    assert envelope['version'] == 4
    assert [item['name'] for item in envelope['sinks']] == ['primary', 'mirror']
    first = await RunTraceLocalSource(tmp_path / 'a').read_events_after(envelope['sinks'][0]['metadata'], 0)
    second = await RunTraceLocalSource(tmp_path / 'b').read_events_after(envelope['sinks'][1]['metadata'], 0)
    assert first == second
    assert await registry.read_result(envelope) == {'success': True}


def test_registry_merges_local_defaults_and_rejects_empty_selection(tmp_path) -> None:
    from nexus.distributed.config import DistributedConfig
    from nexus.distributed.execution.trace_backends import RunTraceSinkRequest, create_run_trace_backend_registry
    from nexus.runner.trace import RunTraceLocalSink

    root = tmp_path / 'override'
    registry = create_run_trace_backend_registry(
        DistributedConfig(run_trace_sinks={'local': {'config': {'root': str(root)}, 'sinkConfig': {'batch_size': 7}}})
    )
    request = RunTraceSinkRequest(execution_id='exec', assignment_id='asn', attempt_id='attempt')
    backends = registry.resolve(run_kind='single', driver_config={}, request=request)
    sink = backends.backend_sinks['local']
    assert isinstance(sink, RunTraceLocalSink)
    assert sink.root == root
    assert sink.batch_size == 7

    with pytest.raises(ValueError, match='at least one sink'):
        registry.resolve(
            run_kind='single',
            driver_config={'run_trace_backends': {'sinks': []}},
            request=request,
        )


def test_registry_adds_artifact_backend_when_internal_credential_is_available(tmp_path) -> None:
    from nexus.distributed.artifacts.run_result import ArtifactTraceSink
    from nexus.distributed.config import DistributedConfig
    from nexus.distributed.execution.trace_backends import RunTraceSinkRequest, create_run_trace_backend_registry

    registry = create_run_trace_backend_registry(
        DistributedConfig(
            secrets={'internal_token': 'i' * 32},
            run_trace_artifact_root=str(tmp_path),
            application_internal_base_url='http://application.internal',
        )
    )
    backends = registry.resolve(
        run_kind='single',
        driver_config={},
        request=RunTraceSinkRequest(execution_id='exec', assignment_id='asn', attempt_id='attempt'),
    )

    assert list(backends.backend_sinks) == ['local', 'artifact']
    artifact_sink = backends.backend_sinks['artifact']
    assert isinstance(artifact_sink, ArtifactTraceSink)
    assert artifact_sink.max_result_size_bytes == 16 * 1024**2
    assert artifact_sink.publish_timeout_seconds == 1800


def test_registry_adds_artifact_only_for_a_platform_managed_fallback(tmp_path) -> None:
    from nexus.distributed.config import DistributedConfig
    from nexus.distributed.execution.trace_backends import RunTraceSinkRequest, create_run_trace_backend_registry

    registry = create_run_trace_backend_registry(
        DistributedConfig(
            secrets={'internal_token': 'i' * 32},
            run_trace_artifact_root=str(tmp_path),
            application_internal_base_url='http://application.internal',
            run_trace_sinks={
                'primary': {
                    'sinkClass': 'nexus.runner.trace.local::RunTraceLocalSink',
                    'sourceClass': 'nexus.runner.trace.local::RunTraceLocalSource',
                    'config': {'root': str(tmp_path / 'primary')},
                }
            },
        )
    )
    user_selection = {
        'run_trace_backends': {
            'sinks': ['primary'],
            'append_artifact_fallback': True,
        }
    }

    with_artifact_scope = registry.resolve(
        run_kind='single',
        driver_config=user_selection,
        request=RunTraceSinkRequest(execution_id='exec', assignment_id='asn', attempt_id='attempt'),
    )
    without_artifact_scope = registry.resolve(
        run_kind='single',
        driver_config=user_selection,
        request=RunTraceSinkRequest(
            execution_id='exec',
            assignment_id='asn',
            attempt_id='attempt',
            artifact_scope_available=False,
        ),
    )
    raw_admin_selection = registry.resolve(
        run_kind='single',
        driver_config={'run_trace_backends': {'sinks': ['primary']}},
        request=RunTraceSinkRequest(execution_id='exec', assignment_id='asn', attempt_id='attempt'),
    )

    assert list(with_artifact_scope.backend_sinks) == ['primary', 'artifact']
    assert list(without_artifact_scope.backend_sinks) == ['primary']
    assert list(raw_admin_selection.backend_sinks) == ['primary']


def test_registry_appends_configured_required_system_backends_without_name_checks(tmp_path) -> None:
    from nexus.distributed.config import DistributedConfig
    from nexus.distributed.execution.trace_backends import RunTraceSinkRequest, create_run_trace_backend_registry

    class_uri = 'nexus.runner.trace.local::RunTraceLocalSink'
    source_uri = 'nexus.runner.trace.local::RunTraceLocalSource'
    registry = create_run_trace_backend_registry(
        DistributedConfig(
            default_run_trace_sinks=['primary'],
            run_trace_sinks={
                'primary': {
                    'sinkClass': class_uri,
                    'sourceClass': source_uri,
                    'config': {'root': str(tmp_path / 'primary')},
                },
                'compliance': {
                    'systemManaged': True,
                    'required': True,
                    'sinkClass': class_uri,
                    'sourceClass': source_uri,
                    'config': {'root': str(tmp_path / 'compliance')},
                },
            },
        )
    )
    request = RunTraceSinkRequest(execution_id='exec', assignment_id='asn', attempt_id='attempt')

    defaulted = registry.resolve(run_kind='single', driver_config={}, request=request)
    selected = registry.resolve(
        run_kind='single',
        driver_config={'run_trace_backends': {'sinks': ['primary'], 'append_artifact_fallback': True}},
        request=request,
    )
    raw_admin_selection = registry.resolve(
        run_kind='single',
        driver_config={'run_trace_backends': {'sinks': ['primary']}},
        request=request,
    )

    assert list(defaulted.backend_sinks) == ['primary', 'compliance']
    assert list(selected.backend_sinks) == ['primary', 'compliance']
    assert list(raw_admin_selection.backend_sinks) == ['primary']


def test_registry_requires_allocation_scope_for_artifact_backend(tmp_path) -> None:
    from nexus.distributed.config import DistributedConfig
    from nexus.distributed.execution.trace_backends import RunTraceSinkRequest, create_run_trace_backend_registry

    registry = create_run_trace_backend_registry(
        DistributedConfig(
            secrets={'internal_token': 'i' * 32},
            run_trace_artifact_root=str(tmp_path),
            application_internal_base_url='http://application.internal',
        )
    )
    request = RunTraceSinkRequest(
        execution_id='exec',
        assignment_id='asn',
        attempt_id='attempt',
        artifact_scope_available=False,
    )

    backends = registry.resolve(run_kind='single', driver_config={}, request=request)
    assert list(backends.backend_sinks) == ['local']

    with pytest.raises(ValueError, match='allocation scope'):
        registry.resolve(
            run_kind='single',
            driver_config={'run_trace_backends': {'sinks': ['artifact']}},
            request=request,
        )


@pytest.mark.asyncio
async def test_registry_dynamically_imports_explicit_sink_and_source_subclasses(monkeypatch, tmp_path) -> None:
    from nexus.distributed.config import DistributedConfig
    from nexus.distributed.execution.trace_backends import RunTraceSinkRequest, create_run_trace_backend_registry

    class ConfiguredSink(RunTraceSink):
        def __init__(self, *, endpoint, component, password, execution_id, assignment_id, attempt_id) -> None:
            self.password = password
            self.values = {
                'endpoint': endpoint,
                'component': component,
                'execution_id': execution_id,
                'assignment_id': assignment_id,
                'attempt_id': attempt_id,
            }

        async def start(self, context) -> None:  # noqa: ANN001
            return None

        async def on_event(self, context, event) -> None:  # noqa: ANN001
            return None

        async def close(self, context) -> None:  # noqa: ANN001
            return None

        def metadata(self):
            return self.values

    class ConfiguredSource(RunTraceSource):
        def __init__(self, *, endpoint, component, password) -> None:
            self.password = password
            self.values = {'endpoint': endpoint, 'component': component}

        async def read_events_after(self, metadata, cursor, *, limit=None):  # noqa: ANN001, ARG002
            return []

        async def read_result(self, metadata):  # noqa: ANN001
            return {**self.values, 'metadata': dict(metadata)}

    module = ModuleType('nexus_test_trace_backend')
    module.ConfiguredSink = ConfiguredSink
    module.ConfiguredSource = ConfiguredSource
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry = create_run_trace_backend_registry(
        DistributedConfig(
            run_trace_artifact_root=str(tmp_path / 'local'),
            default_run_trace_sinks=['custom'],
            secrets={'run_trace_credentials': {'trace-db': 'trace-password'}},
            run_trace_sinks={
                'custom': {
                    'sinkClass': 'nexus_test_trace_backend::ConfiguredSink',
                    'sourceClass': 'nexus_test_trace_backend::ConfiguredSource',
                    'config': {'endpoint': 'memory://trace'},
                    'secretConfig': {'password': 'trace-db'},
                    'sinkConfig': {'component': 'sink'},
                    'sourceConfig': {'component': 'source'},
                }
            },
        )
    )
    backends = registry.resolve(
        run_kind='live',
        driver_config={},
        request=RunTraceSinkRequest(execution_id='exec', assignment_id='asn', attempt_id='attempt'),
    )
    assert backends.backend_sinks['custom'].metadata()['component'] == 'sink'
    assert backends.backend_sinks['custom'].password == 'trace-password'
    envelope = {
        'version': 4,
        'run_id': 'run',
        'run_span_id': 'span',
        'run_kind': 'live',
        'sinks': [{'name': 'custom', 'metadata': {'record': 'result'}}],
    }
    assert await registry.read_result(envelope) == {
        'endpoint': 'memory://trace',
        'component': 'source',
        'metadata': {'record': 'result'},
    }


def test_registry_rejects_missing_or_cleartext_run_trace_secrets(tmp_path) -> None:
    """Misconfigured secrets are still detected, but they fail the run, not the process.

    Registry construction runs inside the instance manager, which hosts unrelated
    executions, so a single bad backend is quarantined rather than raised at start-up.
    """
    from nexus.distributed.config import DistributedConfig
    from nexus.distributed.execution.trace_backends import RunTraceSinkRequest, create_run_trace_backend_registry

    class_uri = 'nexus.runner.trace.local::RunTraceLocalSink'
    source_uri = 'nexus.runner.trace.local::RunTraceLocalSource'
    request = RunTraceSinkRequest(execution_id='exec', assignment_id='asn', attempt_id='attempt')
    select_custom = {'run_trace_backends': {'sinks': ['custom']}}

    registry = create_run_trace_backend_registry(
        DistributedConfig(
            run_trace_artifact_root=str(tmp_path),
            run_trace_sinks={
                'custom': {
                    'sinkClass': class_uri,
                    'sourceClass': source_uri,
                    'secretConfig': {'password': 'missing'},
                }
            },
        )
    )
    with pytest.raises(ValueError, match="run trace credential 'missing' is not configured"):
        registry.resolve(run_kind='single', driver_config=select_custom, request=request)

    with pytest.raises(ValueError, match="sensitive run trace field 'password'.*secretConfig"):
        DistributedConfig(
            run_trace_sinks={
                'custom': {
                    'sinkClass': class_uri,
                    'sourceClass': source_uri,
                    'config': {'password': 'cleartext'},
                }
            },
        )

    # Rejected by the config model itself, before any backend is constructed.
    with pytest.raises(ValueError, match="sensitive run trace field 'password'.*secretConfig"):
        create_run_trace_backend_registry(
            DistributedConfig(
                run_trace_artifact_root=str(tmp_path),
                secrets={'run_trace_credentials': {'trace-db': 'trace-password'}},
                run_trace_sinks={
                    'custom': {
                        'sinkClass': class_uri,
                        'sourceClass': source_uri,
                        'config': {'password': 'cleartext'},
                        'secretConfig': {'password': 'trace-db'},
                    }
                },
            )
        )


def test_tdsql_trace_backend_repr_does_not_include_password() -> None:
    from nexus.runner.trace.tdsql import TDSQLTraceSink, TDSQLTraceSource

    sink = TDSQLTraceSink(host='mysql.example', user='runner', password='trajectory-password', db='traces')
    source = TDSQLTraceSource(host='mysql.example', user='runner', password='trajectory-password', db='traces')

    assert 'trajectory-password' not in repr(sink)
    assert 'trajectory-password' not in repr(source)


def test_registry_rejects_non_subclass_unknown_duplicate_and_reserved_config(tmp_path, monkeypatch) -> None:
    from nexus.distributed.config import DistributedConfig
    from nexus.distributed.execution.trace_backends import RunTraceSinkRequest, create_run_trace_backend_registry

    module = ModuleType('invalid_trace_backend')
    module.Sink = object
    module.Source = object
    monkeypatch.setitem(sys.modules, module.__name__, module)
    request = RunTraceSinkRequest(execution_id='exec', assignment_id='asn', attempt_id='attempt')

    # Both faults below quarantine their backend instead of failing registry construction;
    # the original cause is preserved and surfaces when that backend is selected.
    bad = create_run_trace_backend_registry(
        DistributedConfig(
            run_trace_sinks={
                'bad': {
                    'sinkClass': 'invalid_trace_backend::Sink',
                    'sourceClass': 'invalid_trace_backend::Source',
                }
            }
        )
    )
    with pytest.raises(ValueError, match='must inherit RunTraceSink'):
        bad.resolve(
            run_kind='single',
            driver_config={'run_trace_backends': {'sinks': ['bad']}},
            request=request,
        )

    forged = create_run_trace_backend_registry(
        DistributedConfig(run_trace_sinks={'local': {'sinkConfig': {'execution_id': 'forged'}}})
    )
    with pytest.raises(ValueError, match="reserved key 'execution_id'"):
        forged.resolve(
            run_kind='single',
            driver_config={'run_trace_backends': {'sinks': ['local']}},
            request=request,
        )

    registry = create_run_trace_backend_registry(DistributedConfig(run_trace_artifact_root=str(tmp_path)))
    with pytest.raises(ValueError, match='unknown run trace sink'):
        registry.resolve(
            run_kind='live',
            driver_config={'run_trace_backends': {'sinks': ['missing']}},
            request=request,
        )
    with pytest.raises(ValueError, match='duplicate names'):
        registry.resolve(
            run_kind='live',
            driver_config={'run_trace_backends': {'sinks': ['local', 'local']}},
            request=request,
        )


@pytest.mark.asyncio
async def test_registry_result_reader_falls_back_to_next_source() -> None:
    from nexus.distributed.execution.trace_backends import RunTraceBackendRegistration, RunTraceBackendRegistry

    class Sink(RunTraceSink):
        async def start(self, context) -> None:  # noqa: ANN001
            return None

        async def on_event(self, context, event) -> None:  # noqa: ANN001
            return None

        async def close(self, context) -> None:  # noqa: ANN001
            return None

    class FailingSource(RunTraceSource):
        async def read_events_after(self, metadata, cursor, *, limit=None):  # noqa: ANN001, ARG002
            return []

        async def read_result(self, metadata):  # noqa: ANN001, ARG002
            from nexus.runner.trace import RunTraceSourceUnavailable

            raise RunTraceSourceUnavailable('primary unavailable')

    class GoodSource(FailingSource):
        async def read_result(self, metadata):  # noqa: ANN001
            return {'source': metadata['source']}

    registry = RunTraceBackendRegistry(default_sink_names=['primary', 'fallback'])
    registry.register(RunTraceBackendRegistration('primary', lambda request: Sink(), FailingSource()))
    registry.register(RunTraceBackendRegistration('fallback', lambda request: Sink(), GoodSource()))
    registry.validate()
    envelope = {
        'version': 4,
        'run_id': 'run',
        'run_span_id': 'span',
        'run_kind': 'single',
        'sinks': [
            {'name': 'primary', 'metadata': {'source': 'primary'}},
            {'name': 'fallback', 'metadata': {'source': 'fallback'}},
        ],
    }
    assert await registry.read_result(envelope) == {'source': 'fallback'}


@pytest.mark.asyncio
async def test_registry_does_not_fallback_from_source_validation_errors() -> None:
    from nexus.distributed.execution.trace_backends import RunTraceBackendRegistration, RunTraceBackendRegistry

    class Sink(RunTraceSink):
        async def start(self, context) -> None:  # noqa: ANN001
            return None

        async def on_event(self, context, event) -> None:  # noqa: ANN001
            return None

        async def close(self, context) -> None:  # noqa: ANN001
            return None

    class InvalidSource(RunTraceSource):
        async def read_events_after(self, metadata, cursor, *, limit=None):  # noqa: ANN001, ARG002
            raise ValueError('corrupt events')

        async def read_result(self, metadata):  # noqa: ANN001, ARG002
            raise ValueError('corrupt result')

    class FallbackSource(RunTraceSource):
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def read_events_after(self, metadata, cursor, *, limit=None):  # noqa: ANN001, ARG002
            self.calls.append('events')
            return []

        async def read_result(self, metadata):  # noqa: ANN001, ARG002
            self.calls.append('result')
            return {'source': 'fallback'}

    fallback = FallbackSource()
    registry = RunTraceBackendRegistry(default_sink_names=['primary', 'fallback'])
    registry.register(RunTraceBackendRegistration('primary', lambda request: Sink(), InvalidSource()))
    registry.register(RunTraceBackendRegistration('fallback', lambda request: Sink(), fallback))
    registry.validate()
    envelope = {
        'version': 4,
        'run_id': 'run',
        'run_span_id': 'span',
        'run_kind': 'live',
        'sinks': [
            {'name': 'primary', 'metadata': {}},
            {'name': 'fallback', 'metadata': {}},
        ],
    }

    with pytest.raises(ValueError, match='corrupt result'):
        await registry.read_result(envelope)
    with pytest.raises(ValueError, match='corrupt events'):
        await registry.read_live_events_after(envelope, 0)
    assert fallback.calls == []


@pytest.mark.asyncio
async def test_registry_closes_owned_resources_once() -> None:
    from nexus.distributed.execution.trace_backends import RunTraceBackendRegistry

    class Resource:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    resource = Resource()
    registry = RunTraceBackendRegistry(default_sink_names=())
    registry.add_closeable(resource)

    await registry.close()
    await registry.close()

    assert resource.close_calls == 1


def _incompatible_backend_config(tmp_path, **extra_config):
    """A backend whose config carries a key this build's sink/source cannot accept.

    Mirrors the real incident: a backend is added to the shared config after the running
    code was packaged, so its constructor rejects the new key.
    """
    from nexus.distributed.config import DistributedConfig

    return DistributedConfig(
        run_trace_sinks={
            'local': {'config': {'root': str(tmp_path / 'local')}},
            'future-backend': {
                'sinkClass': 'nexus.runner.trace.local::RunTraceLocalSink',
                'sourceClass': 'nexus.runner.trace.local::RunTraceLocalSource',
                'config': {'root': str(tmp_path / 'future'), 'unknown_future_option': 'boom', **extra_config},
            },
        }
    )


def _missing_class_backend_config(tmp_path):
    """A backend whose sink class this build does not contain at all.

    Distinct from _incompatible_backend_config: that one fails while constructing the
    sink, this one fails earlier, while importing it. Naming a class that only newer
    builds ship is how a new trace flavour is introduced, so the earlier failure has to
    be contained just as well as the later one.
    """
    from nexus.distributed.config import DistributedConfig

    return DistributedConfig(
        run_trace_sinks={
            'local': {'config': {'root': str(tmp_path / 'local')}},
            'future-backend': {
                'sinkClass': 'nexus.runner.trace.local::RunTraceLocalSink',
                'sourceClass': 'nexus.runner.trace.local::ClassFromALaterRelease',
                'config': {'root': str(tmp_path / 'future')},
            },
        }
    )


def test_missing_backend_class_is_quarantined_like_a_bad_config(tmp_path, caplog) -> None:
    import logging

    from nexus.distributed.execution.trace_backends import RunTraceSinkRequest, create_run_trace_backend_registry
    from nexus.runner.trace import RunTraceLocalSink

    with caplog.at_level(logging.WARNING):
        registry = create_run_trace_backend_registry(_missing_class_backend_config(tmp_path))

    request = RunTraceSinkRequest(execution_id='exec', assignment_id='asn', attempt_id='attempt')
    backends = registry.resolve(run_kind='single', driver_config={}, request=request)

    assert isinstance(backends.backend_sinks['local'], RunTraceLocalSink)
    assert 'future-backend' not in backends.backend_sinks
    records = [record for record in caplog.records if 'future-backend' in record.getMessage()]
    assert records and {record.levelno for record in records} == {logging.WARNING}
    assert any('failed to import' in record.getMessage() for record in records)


def test_missing_backend_class_still_fails_runs_that_request_it(tmp_path) -> None:
    from nexus.distributed.execution.trace_backends import RunTraceSinkRequest, create_run_trace_backend_registry

    registry = create_run_trace_backend_registry(_missing_class_backend_config(tmp_path))
    request = RunTraceSinkRequest(execution_id='exec', assignment_id='asn', attempt_id='attempt')

    with pytest.raises(ValueError, match='not usable by this build'):
        registry.resolve(
            run_kind='single',
            driver_config={'run_trace_backends': {'sinks': ['future-backend']}},
            request=request,
        )


def test_unusable_backend_does_not_block_registry_construction(tmp_path, caplog) -> None:
    """An unrelated execution must still start when one backend is unusable."""
    import logging

    from nexus.distributed.execution.trace_backends import RunTraceSinkRequest, create_run_trace_backend_registry
    from nexus.runner.trace import RunTraceLocalSink

    with caplog.at_level(logging.WARNING):
        registry = create_run_trace_backend_registry(_incompatible_backend_config(tmp_path))

    request = RunTraceSinkRequest(execution_id='exec', assignment_id='asn', attempt_id='attempt')
    backends = registry.resolve(run_kind='single', driver_config={}, request=request)

    # The default selection silently skips the unusable backend instead of crash-looping.
    assert isinstance(backends.backend_sinks['local'], RunTraceLocalSink)
    assert 'future-backend' not in backends.backend_sinks

    # Warning, not error: a build predating a backend is expected across a mixed-version
    # fleet, so this must not page anyone. The loud signal lives at selection instead
    # (see test_unusable_backend_still_fails_runs_that_request_it).
    records = [record for record in caplog.records if 'future-backend' in record.getMessage()]
    assert records, 'the quarantined backend must still be reported'
    assert {record.levelno for record in records} == {logging.WARNING}


def test_unusable_backend_still_fails_runs_that_request_it(tmp_path) -> None:
    """Fail-fast is preserved where it is attributable: the run that asked for it."""
    from nexus.distributed.execution.trace_backends import RunTraceSinkRequest, create_run_trace_backend_registry

    registry = create_run_trace_backend_registry(_incompatible_backend_config(tmp_path))
    request = RunTraceSinkRequest(execution_id='exec', assignment_id='asn', attempt_id='attempt')

    with pytest.raises(ValueError, match='not usable by this build'):
        registry.resolve(
            run_kind='single',
            driver_config={'run_trace_backends': {'sinks': ['future-backend']}},
            request=request,
        )


@pytest.mark.asyncio
async def test_unusable_backend_source_reports_unavailable_for_fallback(tmp_path) -> None:
    """Reads fall through to a healthy backend rather than failing the whole read."""
    from nexus.distributed.execution.trace_backends import create_run_trace_backend_registry
    from nexus.runner.trace import RunTraceSourceUnavailable

    registry = create_run_trace_backend_registry(_incompatible_backend_config(tmp_path))
    source = registry._registrations['future-backend'].source

    with pytest.raises(RunTraceSourceUnavailable, match='future-backend'):
        await source.read_result({})
    with pytest.raises(RunTraceSourceUnavailable, match='future-backend'):
        await source.read_events_after({}, 0)


def test_unusable_backend_from_unimportable_class_is_also_tolerated(tmp_path) -> None:
    """The same degradation covers a sinkClass this build does not ship at all."""
    from nexus.distributed.config import DistributedConfig
    from nexus.distributed.execution.trace_backends import RunTraceSinkRequest, create_run_trace_backend_registry

    registry = create_run_trace_backend_registry(
        DistributedConfig(
            run_trace_sinks={
                'local': {'config': {'root': str(tmp_path / 'local')}},
                'not-shipped': {
                    'sinkClass': 'nexus.runner.trace.does_not_exist::FutureSink',
                    'sourceClass': 'nexus.runner.trace.does_not_exist::FutureSource',
                },
            }
        )
    )
    request = RunTraceSinkRequest(execution_id='exec', assignment_id='asn', attempt_id='attempt')

    assert 'not-shipped' not in registry.resolve(
        run_kind='single', driver_config={}, request=request
    ).backend_sinks
