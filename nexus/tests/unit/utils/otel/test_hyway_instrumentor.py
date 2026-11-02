from __future__ import annotations

import asyncio
import os
import subprocess
import sys

from datetime import datetime
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from nexus.distributed.repositories.execution import CreateExecution, ExecutionRepository
from nexus.utils.otel.propagation import inject_carrier, use_remote_context


PARENT = {
    'traceparent': '00-11111111111111111111111111111111-2222222222222222-01',
    'tracestate': 'vendor=preserved',
}


def _row(request):
    return ExecutionRepository(None, shard_count=8)._new_execution_row(
        request, placement={}, spec_sha256='test', now=datetime(2026, 1, 1)
    )


def _request(**kwargs):
    return CreateExecution('exe', 'job', 'nexus_single', 'asyncio', kwargs.pop('spec', {}), **kwargs)


def test_entry_points_only_register_cloud_and_subprocess():
    registered = {ep.name: ep.value for ep in entry_points(group='opentelemetry_instrumentor')}
    assert registered.get('nexus-hyway') == 'nexus.utils.otel.hyway:NexusHywayInstrumentor'
    assert 'nexus-submit' not in registered
    assert 'nexus-execution' not in registered


def test_constructor_captures_carrier_without_instrumentation(monkeypatch):
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    with use_remote_context(PARENT, fallback_only=False):
        assert _request().trace_carrier == PARENT
        assert _request(trace_carrier=None).trace_carrier is None
        assert _request(trace_carrier=PARENT).trace_carrier is PARENT


def test_cloud_injects_slice_only_when_building_row(otel_hyway_instrumentor):
    with use_remote_context(PARENT, fallback_only=False):
        request = _request(idempotency_key='submission-1')
        assert request.trace_carrier == PARENT
        row = _row(request)
        assert row.trace_carrier_json == {**PARENT, 'tracestate': 'hyway_slice_id=submission-1,vendor=preserved'}
        assert request.trace_carrier == PARENT
        assert inject_carrier() == PARENT


def test_batch_child_keeps_parent_slice_and_adds_item(otel_hyway_instrumentor):
    carrier = {**PARENT, 'tracestate': 'hyway_slice_id=parent-request,vendor=preserved'}
    request = _request(
        trace_carrier=carrier,
        idempotency_key='batch:parent:item:item-1:attempt:2',
        parent_execution_id='parent',
        spec={'batch': {'parent_execution_id': 'parent', 'item_id': 'item-1', 'attempt_no': 2}},
    )
    row = _row(request)
    assert row.trace_carrier_json == {
        **PARENT,
        'tracestate': 'hyway_instance_id=item-1,hyway_slice_id=parent-request,vendor=preserved',
    }
    assert request.trace_carrier is carrier


def test_child_without_parent_slice_does_not_use_retry_key(otel_hyway_instrumentor):
    request = _request(trace_carrier=PARENT, idempotency_key='retry-key', spec={'batch': {'item_id': 'item-1'}})
    assert _row(request).trace_carrier_json['tracestate'] == 'hyway_instance_id=item-1,vendor=preserved'


def test_top_level_batch_gets_submission_slice(otel_hyway_instrumentor):
    request = _request(trace_carrier=PARENT, idempotency_key='batch-submit', spec={'batch': {'retry_failed': 2}})
    assert _row(request).trace_carrier_json['tracestate'] == 'hyway_slice_id=batch-submit,vendor=preserved'


def test_existing_members_and_explicit_carrier_take_precedence(otel_hyway_instrumentor):
    state = 'hyway_slice_id=upstream,hyway_instance_id=upstream-item,vendor=preserved'
    carrier = {**PARENT, 'tracestate': state}
    with use_remote_context(PARENT, fallback_only=False):
        for spec in ({}, {'batch': {'item_id': 'new-item'}}):
            row = _row(_request(trace_carrier=carrier, idempotency_key='new-key', spec=spec))
            assert row.trace_carrier_json == carrier
        assert _row(_request(trace_carrier=None, idempotency_key='request')).trace_carrier_json is None


async def test_concurrent_cloud_rows_keep_separate_carriers(otel_hyway_instrumentor):
    async def create(key):
        with use_remote_context(PARENT, fallback_only=False):
            request = _request(idempotency_key=key)
            await asyncio.sleep(0)
            return _row(request).trace_carrier_json

    rows = await asyncio.gather(create('first'), create('second'))
    assert [row['tracestate'] for row in rows] == [
        'hyway_slice_id=first,vendor=preserved',
        'hyway_slice_id=second,vendor=preserved',
    ]


def test_disabled_hook_preserves_persistence(monkeypatch, otel_hyway_instrumentor):
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'false')
    assert _row(_request(trace_carrier=PARENT, idempotency_key='request')).trace_carrier_json == PARENT


def test_injection_failure_preserves_row(monkeypatch, otel_hyway_instrumentor):
    from nexus.utils.otel import hyway

    def fail(*args):
        raise RuntimeError('carrier failed')

    monkeypatch.setattr(hyway, 'add_trace_state_entries', fail)
    row = _row(_request(trace_carrier=PARENT, idempotency_key='request'))
    assert row.execution_id == 'exe'
    assert row.trace_carrier_json == PARENT


def test_uninstrument_restores_row_factory(otel_hyway_instrumentor):
    from nexus.utils.otel.hyway import NexusHywayInstrumentor

    original = ExecutionRepository._new_execution_row.__wrapped__
    assert NexusHywayInstrumentor() is otel_hyway_instrumentor
    otel_hyway_instrumentor.instrument()
    otel_hyway_instrumentor.uninstrument()
    assert ExecutionRepository._new_execution_row is original
    assert _row(_request(trace_carrier=PARENT, idempotency_key='request')).trace_carrier_json == PARENT


@pytest.mark.parametrize('bootstrap', ['nexus', 'programmatic', 'official'])
@pytest.mark.parametrize('disabled', ['', 'nexus-hyway', '*'])
def test_startup_only_patches_cloud_row_factory(bootstrap, disabled):
    script = """
import os, sys
if sys.argv[1] == 'programmatic':
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    trace.set_tracer_provider(TracerProvider())
import nexus
assert 'nexus.cli.commands.submit' not in sys.modules
assert 'nexus.distributed.repositories.execution' not in sys.modules
from nexus.cli.commands import submit
from nexus.distributed.repositories.execution import CreateExecution, ExecutionRepository
expected = not bool(os.environ['OTEL_PYTHON_DISABLED_INSTRUMENTATIONS'])
assert not hasattr(submit._submit, '__wrapped__')
assert not hasattr(submit._submission_payload, '__wrapped__')
assert not hasattr(CreateExecution.__init__, '__wrapped__')
assert hasattr(ExecutionRepository._new_execution_row, '__wrapped__') == expected
"""
    env = {
        **os.environ,
        'NEXUS_OTEL_ENABLED': 'true',
        'OTEL_TRACES_EXPORTER': 'none',
        'OTEL_METRICS_EXPORTER': 'none',
        'OTEL_LOGS_EXPORTER': 'none',
        'OTEL_PYTHON_DISABLED_INSTRUMENTATIONS': disabled,
    }
    command = [sys.executable, '-c', script, bootstrap]
    if bootstrap == 'official':
        command.insert(0, str(Path(sys.executable).with_name('opentelemetry-instrument')))
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Attempting to instrument while already instrumented' not in result.stderr


def test_cancelled_deferred_hook_does_not_patch_after_import():
    script = """
import sys
import nexus
from nexus.utils.otel.hyway import NexusHywayInstrumentor
assert 'nexus.distributed.repositories.execution' not in sys.modules
instrumentor = NexusHywayInstrumentor()
instrumentor.instrument()
instrumentor.uninstrument()
from nexus.distributed.repositories.execution import ExecutionRepository
original = ExecutionRepository._new_execution_row
assert not hasattr(original, '__wrapped__')
instrumentor.instrument()
assert ExecutionRepository._new_execution_row is not original
instrumentor.uninstrument()
assert ExecutionRepository._new_execution_row is original
"""
    env = {
        **os.environ,
        'NEXUS_OTEL_ENABLED': 'true',
        'OTEL_TRACES_EXPORTER': 'none',
        'OTEL_METRICS_EXPORTER': 'none',
        'OTEL_LOGS_EXPORTER': 'none',
        'OTEL_PYTHON_DISABLED_INSTRUMENTATIONS': '*',
    }
    result = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
