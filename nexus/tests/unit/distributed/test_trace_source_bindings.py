from __future__ import annotations

import copy

from datetime import datetime, timezone

import pytest

from nexus.distributed.errors import DomainConflictError, FencingError
from nexus.distributed.execution.drivers import nexus_single
from nexus.distributed.execution.executors.base import ExecutionHandle
from nexus.distributed.execution.runtime_info import merge_runtime_info
from nexus.distributed.execution.trace_backends import RunTraceBackendRegistration, RunTraceBackendRegistry
from nexus.distributed.persistence.orm import AssignmentRow, ExecutionRow, InstanceManagerRow
from nexus.distributed.repositories.manager import ManagerRepository
from nexus.distributed.trace.bindings import (
    TraceSourceIdentity,
    make_source_binding,
    source_binding,
    validate_source_binding_update,
)
from nexus.distributed.utils.clock import utcnow
from nexus.runner.single import RunOutput
from nexus.runner.trace import RunTraceLocalSink, RunTraceLocalSource, principal_trace_scope_id
from tests.unit.distributed.drivers.test_drivers import CallbackRecorder
from tests.unit.distributed.test_repositories import _assign_execution, _register_manager
from tests.unit.distributed.test_repositories import session_factory as session_factory


def _binding(**kwargs):
    return make_source_binding(
        run_id=kwargs.get('run_id', 'exec'),
        run_span_id=kwargs.get('run_span_id', 'span'),
        assignment_id=kwargs.get('assignment_id', 'asn'),
        attempt_id=kwargs.get('attempt_id', 'attempt'),
        owner_principal_id='legacy',
        selected_sinks=('tdsql-hyway', 'artifact'),
        source_identities={'tdsql-hyway': TraceSourceIdentity(source_id='hyway-dataset', source_revision='1')},
    )


def _runtime(binding):
    return {'run_trace': {'source_binding': binding}}


def _validate(existing, incoming):
    validate_source_binding_update(
        existing,
        incoming,
        execution_id='exec',
        assignment_id='asn',
        attempt_id='attempt',
        owner_principal_id='legacy',
    )


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('run_id', 'other'),
        ('run_span_id', 'other'),
        ('assignment_id', 'other'),
        ('attempt_id', 'other'),
        ('sources', []),
        ('selected_sinks', ['artifact']),
    ],
)
def test_binding_cannot_be_replaced(field, value):
    binding = _binding()
    replacement = copy.deepcopy(binding)
    replacement[field] = value
    with pytest.raises(DomainConflictError):
        _validate(_runtime(binding), _runtime(replacement))


def test_binding_scope_uses_owner_and_rejects_unknown_fields():
    binding = _binding()
    assert binding['sources'][0]['principal_scope_id'] == principal_trace_scope_id('legacy')
    binding['sources'][0]['principal_scope_id'] = principal_trace_scope_id('other')
    with pytest.raises(DomainConflictError, match='scope_mismatch'):
        _validate({}, _runtime(binding))
    binding = _binding()
    binding['sources'][0]['password'] = 'never-report-this'
    with pytest.raises(DomainConflictError, match='binding_invalid') as error:
        source_binding(_runtime(binding))
    assert 'never-report-this' not in str(error.value)


def test_final_metadata_preserves_binding_and_checks_explicit_identity():
    binding = _binding()
    final = {
        '_artifacts': {
            'single': {
                'run_id': 'exec',
                'run_span_id': 'span',
                'run_kind': 'single',
                'sinks': [{'name': 'tdsql-hyway', 'metadata': {}}, {'name': 'artifact', 'metadata': {}}],
            }
        }
    }
    _validate(_runtime(binding), final)
    assert source_binding(merge_runtime_info(_runtime(binding), final)) == source_binding(_runtime(binding))
    final['_artifacts']['single']['sinks'][0]['metadata']['principal_scope_id'] = '0' * 32
    with pytest.raises(DomainConflictError, match='artifact_mismatch'):
        _validate(_runtime(binding), final)


@pytest.mark.parametrize('explicit', [False, True])
async def test_single_reports_actual_source_before_task_and_before_close(monkeypatch, tmp_path, explicit):
    registry = RunTraceBackendRegistry(default_sink_names=('tdsql-hyway',))
    registry.register(
        RunTraceBackendRegistration(
            name='tdsql-hyway',
            sink_factory=lambda request: RunTraceLocalSink(
                root=tmp_path,
                execution_id=request.execution_id,
                assignment_id=request.assignment_id,
                attempt_id=request.attempt_id,
            ),
            source=RunTraceLocalSource(root=tmp_path),
            source_identity=TraceSourceIdentity(source_id='actual-hyway', source_revision='7'),
        )
    )
    callbacks = CallbackRecorder()
    driver = nexus_single.NexusSingleDriver(
        execution_id='exec',
        assignment_id='asn',
        attempt_id='attempt',
        owner_principal_id='legacy',
        trace_registry=registry,
        artifact_scope_available=False,
    )

    async def run(run_input, hooks):
        assert callbacks.running.is_set()
        binding = source_binding(callbacks.running_payload['runtime_info'])
        assert binding is not None
        assert binding.selected_sinks == ('tdsql-hyway',)
        assert binding.sources[0].source_id == 'actual-hyway'
        assert binding.sources[0].source_revision == '7'
        assert binding.run_span_id == driver._trace_session.run_span_id
        with pytest.raises(RuntimeError, match='before close'):
            driver._trace_session.artifact_metadata()
        output = RunOutput(success=True, task_output={'ok': True})
        await hooks.on_run_output(None, output)
        return output

    monkeypatch.setattr(nexus_single, 'run_single', run)
    config = {'run_trace_backends': {'sinks': ['tdsql-hyway']}} if explicit else {}
    await driver.start(
        ExecutionHandle(execution_id='exec', assignment_id='asn'), {'task_name': 'identity'}, config, callbacks
    )
    assert callbacks.completed.is_set(), callbacks.error
    assert not callbacks.failed.is_set()


async def test_single_reports_the_partition_key_with_the_running_state(monkeypatch, tmp_path):
    """The Console reads a running Execution from the binding, never from _artifacts.

    _artifacts is published when the run closes, so a partition key that only travelled
    there made every running Boundless read fail resolution.
    """
    registry = RunTraceBackendRegistry(default_sink_names=('tdsql-hyway',))
    registry.register(
        RunTraceBackendRegistration(
            name='tdsql-hyway',
            sink_factory=lambda request: RunTraceLocalSink(
                root=tmp_path,
                execution_id=request.execution_id,
                assignment_id=request.assignment_id,
                attempt_id=request.attempt_id,
            ),
            source=RunTraceLocalSource(root=tmp_path),
            source_identity=TraceSourceIdentity(source_id='actual-hyway', source_revision='7'),
        )
    )
    callbacks = CallbackRecorder()
    created_at = datetime(2026, 9, 20, 10, 11, 12, 131415, tzinfo=timezone.utc)
    driver = nexus_single.NexusSingleDriver(
        execution_id='exec',
        assignment_id='asn',
        attempt_id='attempt',
        owner_principal_id='legacy',
        trace_registry=registry,
        artifact_scope_available=False,
        execution_created_at=created_at,
    )

    async def run(run_input, hooks):
        binding = source_binding(callbacks.running_payload['runtime_info'])
        assert binding is not None
        # Normalized to the MySQL literal the sink envelope will publish, not the ISO form.
        assert binding.parent_created_at == '2026-09-20 10:11:12.131415'
        output = RunOutput(success=True, task_output={'ok': True})
        await hooks.on_run_output(None, output)
        return output

    monkeypatch.setattr(nexus_single, 'run_single', run)
    await driver.start(
        ExecutionHandle(execution_id='exec', assignment_id='asn'), {'task_name': 'identity'}, {}, callbacks
    )
    assert callbacks.completed.is_set(), callbacks.error


async def test_repository_binding_idempotence_and_terminal_preservation(session_factory):
    await _register_manager(session_factory, manager_id='im-1')
    execution, assignment = await _assign_execution(session_factory, 'bound-single')
    binding = _binding(
        run_id=execution.execution_id, assignment_id=assignment.assignment_id, attempt_id=assignment.attempt_id
    )
    async with session_factory() as session:
        repo = ManagerRepository(session)
        await repo.ack_assignment(assignment.assignment_id, 'im-1', assignment.manager_session_id, now=utcnow())
        for _ in range(2):
            await repo.mark_running(
                assignment.assignment_id,
                'im-1',
                assignment.manager_session_id,
                endpoint_url=None,
                endpoint_ready=False,
                runtime_info=_runtime(binding),
                now=utcnow(),
            )
        await repo.fail_execution(
            assignment.assignment_id,
            'im-1',
            assignment.manager_session_id,
            error={'type': 'fixture'},
            runtime_info={'run_trace': {'diagnostic': 'closed'}},
            now=utcnow(),
        )
        await session.commit()
    async with session_factory() as session:
        row = await session.get(AssignmentRow, assignment.assignment_id)
        assert source_binding(row.runtime_info_json) == source_binding(_runtime(binding))
        assert row.runtime_info_json['run_trace']['diagnostic'] == 'closed'


@pytest.mark.parametrize('mismatch', ['manager', 'assignment', 'attempt', 'reported_assignment'])
async def test_repository_rejects_stale_binding_report(session_factory, mismatch):
    await _register_manager(session_factory, manager_id='im-1')
    execution, assignment = await _assign_execution(session_factory, 'fenced-single')
    binding = _binding(
        run_id=execution.execution_id, assignment_id=assignment.assignment_id, attempt_id=assignment.attempt_id
    )
    async with session_factory() as session:
        repo = ManagerRepository(session)
        await repo.ack_assignment(assignment.assignment_id, 'im-1', assignment.manager_session_id, now=utcnow())
        if mismatch == 'manager':
            row = await session.get(InstanceManagerRow, 'im-1')
            row.session_id = 'replacement-session'
        elif mismatch in {'assignment', 'attempt'}:
            row = await session.get(ExecutionRow, execution.execution_id)
            setattr(row, f'current_{mismatch}_id', 'replacement')
        else:
            binding['assignment_id'] = 'wrong-assignment'
        await session.commit()
    async with session_factory() as session:
        with pytest.raises((DomainConflictError, FencingError)):
            await ManagerRepository(session).mark_running(
                assignment.assignment_id,
                'im-1',
                assignment.manager_session_id,
                endpoint_url=None,
                endpoint_ready=False,
                runtime_info=_runtime(binding),
                now=utcnow(),
            )
        await session.rollback()
    async with session_factory() as session:
        row = await session.get(AssignmentRow, assignment.assignment_id)
        assert source_binding(row.runtime_info_json) is None
