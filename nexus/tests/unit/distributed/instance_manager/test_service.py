from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import stat
import struct
import threading

from datetime import timedelta
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest

from sqlalchemy import event, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as SqlAlchemyTimeoutError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nexus.dataset.task_files import LocalZipTaskFiles, task_files_from_spec
from nexus.dataset.zip_index import write_nexus_zip_directory_to_file
from nexus.distributed.batch.data_list import DataListWriter, read_indexed_frame, scan_data_list
from nexus.distributed.batch.task_store import BatchTaskRecord
from nexus.distributed.config import DistributedConfig
from nexus.distributed.execution.executors.asyncio_executor import AsyncioExecutor
from nexus.distributed.execution.executors.base import (
    ExecutionDriver,
    ExecutionHandle,
    ExecutorCapacitySnapshot,
    ExecutorHealthSnapshot,
    QuarantinedAssignment,
    StopDisposition,
)
from nexus.distributed.execution.executors.thread_executor import ThreadExecutor
from nexus.distributed.execution.models import (
    AssignmentRecord,
    AssignmentStatus,
    CommandStatus,
    ExecutionMode,
    ExecutionRecord,
    ExecutionStatus,
    RouteStatus,
)
from nexus.distributed.instance_manager.reporting import (
    ManagerBatchControl,
    ManagerExecutionReporter,
    RunningAssignment,
    TemporaryTaskFiles,
    retry_transient_db_transaction,
)
from nexus.distributed.instance_manager.self_fence import HardExitGuard
from nexus.distributed.instance_manager.service import InstanceManagerService
from nexus.distributed.observability import logging as distributed_logs
from nexus.distributed.persistence.orm import (
    AssignmentRow,
    ExecutionCommandRow,
    ExecutionRow,
    InstanceManagerLivenessRow,
    InstanceManagerRow,
    create_schema,
)
from nexus.distributed.repositories.execution import CreateExecution, ExecutionRepository
from nexus.distributed.repositories.gateway import GatewayRepository
from nexus.distributed.repositories.manager import ManagerRepository
from nexus.distributed.repositories.scheduler import SchedulerRepository
from nexus.distributed.utils.clock import utcnow


@pytest.fixture
async def session_factory():
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    await create_schema(engine)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


def _v3_task_record(
    tmp_path: Path,
    *,
    parent_execution_id: str,
    item_id: str,
    content: str,
) -> BatchTaskRecord:
    task_files_root = tmp_path / f'{item_id}-task-files'
    task_files_root.mkdir()
    (task_files_root / 'input.txt').write_text(content)
    zip_path = tmp_path / f'{item_id}.zip'
    write_nexus_zip_directory_to_file(task_files_root, zip_path)
    data_list_path = tmp_path / f'{item_id}.data_list.bin'
    with DataListWriter(data_list_path) as writer:
        data_ref = writer.write_frame(item_id=item_id, zip_path=zip_path)
    manifest = writer.result()
    index = scan_data_list(
        data_list_path,
        max_frame_bytes=data_ref.length,
        max_bytes=manifest.size_bytes,
        max_items=1,
    )
    frame = read_indexed_frame(
        data_list_path,
        data_ref,
        expected_frame=index.frames[0],
        expected_item_id=item_id,
    )
    return BatchTaskRecord.from_v3_identity(
        parent_execution_id=parent_execution_id,
        identity={
            'version': 3,
            'item_id': item_id,
            'run_input': {'task_name': 'identity'},
            'metadata': {'source_item_id': item_id},
            'data_ref': data_ref.model_dump(mode='json'),
        },
        frame=frame,
        max_record_bytes=256 * 1024**2,
    )


def _malformed_v3_task_record(
    tmp_path: Path,
    *,
    parent_execution_id: str,
    item_id: str,
    mutation: str,
) -> BatchTaskRecord:
    if mutation == 'zip':
        zip_bytes = b'not-a-nexus-zip'
        metadata = {
            'item_id': item_id,
            'task_files': {
                'type': 'zip',
                'sha256': hashlib.sha256(zip_bytes).hexdigest(),
                'size_bytes': len(zip_bytes),
            },
            'version': 1,
        }
        metadata_bytes = json.dumps(metadata, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode()
        frame = struct.pack('<QQQ', 24 + len(metadata_bytes) + len(zip_bytes), len(metadata_bytes), len(zip_bytes))
        frame += metadata_bytes + zip_bytes
        data_ref = {
            'version': 1,
            'offset': 32,
            'length': len(frame),
            'sha256': hashlib.sha256(frame).hexdigest(),
        }
        identity = {
            'version': 3,
            'item_id': item_id,
            'run_input': {'task_name': 'identity'},
            'metadata': {'source_item_id': item_id},
            'data_ref': data_ref,
        }
        identity_bytes = json.dumps(identity, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode()
        return BatchTaskRecord(
            parent_execution_id=parent_execution_id,
            item_id=item_id,
            record_version=3,
            record_sha256=hashlib.sha256(identity_bytes).hexdigest(),
            record_data=struct.pack('<4sHHQQ', b'NXBR', 3, 0, len(identity_bytes), len(frame)) + identity_bytes + frame,
        )

    valid = _v3_task_record(
        tmp_path,
        parent_execution_id=parent_execution_id,
        item_id=item_id,
        content='malformed record payload',
    )
    record_data = bytearray(valid.record_data)
    if mutation == 'nxbr':
        record_data[:4] = b'FAIL'
    elif mutation == 'frame':
        record_data[-1] ^= 0xFF
    else:
        raise AssertionError(f'unsupported malformed record mutation: {mutation}')
    return BatchTaskRecord(
        parent_execution_id=valid.parent_execution_id,
        item_id=valid.item_id,
        record_version=valid.record_version,
        record_sha256=valid.record_sha256,
        record_data=bytes(record_data),
    )


def _task_store_child_execution(
    *,
    parent_execution_id: str,
    child_execution_id: str,
    assignment_id: str,
    record: BatchTaskRecord,
) -> ExecutionRecord:
    now = utcnow()
    return ExecutionRecord(
        execution_id=child_execution_id,
        shard_id=1,
        mode=ExecutionMode.JOB,
        spec_type='nexus_single',
        instance_mode='asyncio',
        spec={
            'spec_version': 1,
            'run_input_ref': {
                'version': 1,
                'store': 'tdsql',
                'parent_execution_id': parent_execution_id,
                'item_id': record.item_id,
                'record_sha256': record.record_sha256,
            },
            'batch': {'parent_execution_id': parent_execution_id, 'item_id': record.item_id, 'attempt_no': 0},
        },
        driver_config={},
        placement={},
        status=ExecutionStatus.ASSIGNED,
        route_status=RouteStatus.NONE,
        slot_cost=1,
        current_assignment_id=assignment_id,
        current_attempt_id='attempt-1',
        attempt_count=0,
        max_attempts=1,
        next_schedule_at=now,
        created_at=now,
        updated_at=now,
        parent_execution_id=parent_execution_id,
    )


def _temporary_task_files(tmp_path: Path, assignment_id: str) -> TemporaryTaskFiles:
    directory = tmp_path / f'temporary-{assignment_id}'
    directory.mkdir()
    zip_path = directory / 'files.zip'
    zip_path.write_bytes(b'test zip placeholder')
    return TemporaryTaskFiles(directory=directory, zip_path=zip_path)


async def test_manager_reporter_forwards_terminal_runtime_info_in_one_call():
    now = utcnow()
    assignment = AssignmentRecord(
        assignment_id='asn-terminal-runtime-info',
        execution_id='exec-terminal-runtime-info',
        attempt_id='attempt-terminal-runtime-info',
        attempt_seq=1,
        manager_id='im-terminal-runtime-info',
        manager_session_id='session-terminal-runtime-info',
        status=AssignmentStatus.ACTIVE,
        slot_cost=1,
        created_at=now,
        updated_at=now,
    )
    calls: list[dict] = []

    class Service:
        async def call_manager_loop(self, operation, *, timeout=30.0):
            del timeout
            return await operation

        async def _complete(self, assignment_id, *, result, runtime_info):
            calls.append({
                'assignment_id': assignment_id,
                'result': result,
                'runtime_info': runtime_info,
            })

    reporter = ManagerExecutionReporter(cast(InstanceManagerService, Service()), assignment)
    runtime_info = {'_artifacts': {'single': {'version': 4}}}

    await reporter.complete({'success': True}, runtime_info=runtime_info)

    assert calls == [
        {
            'assignment_id': assignment.assignment_id,
            'result': {'success': True},
            'runtime_info': runtime_info,
        }
    ]


async def test_legacy_manager_without_workload_is_execution_only(session_factory):
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(aiomonitor_enabled=False),
        manager_id='legacy-manager',
        endpoint_url='http://legacy-manager',
        capacity_total=1,
    )
    try:
        assert service.supported_spec_types == ['nexus_live', 'nexus_single']
    finally:
        await service.batch_control.close()


async def test_instance_manager_shares_and_closes_artifact_transport(session_factory):
    config = DistributedConfig(aiomonitor_enabled=False, secrets={'internal_token': 'internal-token'})
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='artifact-manager',
        endpoint_url='http://artifact-manager',
        capacity_total=1,
    )
    transport = service._artifact_transport

    assert transport is not None
    assert service.batch_control._artifact_transport is transport
    assert service.trace_registry.artifact_transport is transport

    await transport._http_runtime.run(lambda _session: asyncio.sleep(0))
    await service.stop()

    assert transport.session is not None
    assert transport.session.closed


async def test_instance_manager_registers_allocation_identity_and_capability_snapshot(session_factory):
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(aiomonitor_enabled=False),
        manager_id='allocation-a-0',
        endpoint_url='http://allocation-a-0',
        capacity_total=20,
        owner_principal_id='principal-a',
        environment_id='environment-a',
        pool_id='execution',
        allocation_id='allocation-a',
        workload='execution',
        allocation_spec_hash='a' * 64,
        capability_snapshot={'python_version': '3.13', 'bootstrap': 1},
        metadata={'endpoint_kind': 'pod_ip', 'pod_uid': 'pod-uid-a'},
    )
    try:
        assert await service._register_session()
        async with session_factory() as session:
            row = await session.get(InstanceManagerRow, 'allocation-a-0')
        assert row.environment_id == 'environment-a'
        assert row.pool_id == 'execution'
        assert row.allocation_id == 'allocation-a'
        assert row.workload == 'execution'
        assert row.allocation_spec_hash == 'a' * 64
        assert row.capability_snapshot_json == {'python_version': '3.13', 'bootstrap': 1}
        assert row.metadata_json == {'endpoint_kind': 'pod_ip', 'pod_uid': 'pod-uid-a'}
    finally:
        await service.batch_control.close()


class CompletingDriver:
    stop_disposition = StopDisposition.IMMEDIATE

    async def start(self, handle, spec, config, callbacks):
        await callbacks.mark_running(endpoint_url=None, endpoint_ready=False, runtime_info={})
        await callbacks.complete({'ok': True})

    async def submit_command(self, handle, command_type, payload):
        raise AssertionError('not used')

    async def stop(self, handle):
        pass


class FailingWithResultDriver:
    stop_disposition = StopDisposition.IMMEDIATE

    async def start(self, handle, spec, config, callbacks):  # noqa: ARG002
        await callbacks.mark_running(endpoint_url=None, endpoint_ready=False, runtime_info={})
        await callbacks.fail(
            {'type': 'TaskFailed', 'message': 'nope'},
            result={'_external': True, 'kind': 'nexus_run_trace_result', 'metadata': {'version': 4}},
        )

    async def submit_command(self, handle, command_type, payload):  # noqa: ARG002
        raise AssertionError('not used')

    async def stop(self, handle):  # noqa: ARG002
        pass


class UnhandledFailureDriver:
    stop_disposition = StopDisposition.IMMEDIATE

    async def start(self, handle, spec, config, callbacks):  # noqa: ARG002
        raise RuntimeError('driver escaped its exception boundary')

    async def submit_command(self, handle, command_type, payload):  # noqa: ARG002
        raise AssertionError('not used')

    async def stop(self, handle):  # noqa: ARG002
        pass


async def test_batch_control_calls_are_marshaled_to_manager_loop(session_factory, tmp_path):
    manager_loop = asyncio.get_running_loop()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(
            aiomonitor_enabled=False,
            artifact_local_root=str(tmp_path / 'artifacts'),
        ),
        manager_id='batch-loop-manager',
        endpoint_url='http://batch-loop-manager',
        capacity_total=1,
        workload='orchestration',
    )
    original_batch_control = service.batch_control

    class LoopRecordingBatchControl:
        async def prepare(self, parent_execution_id, *, parent_attempt_id=None):
            return parent_execution_id, parent_attempt_id, asyncio.get_running_loop()

    service.batch_control = LoopRecordingBatchControl()
    service.manager_loop = manager_loop
    await service.executor.start()
    try:
        result = await service.executor._run(
            lambda: ManagerBatchControl(service).prepare('batch-1', parent_attempt_id='attempt-1')
        )
    finally:
        await service.executor.close()
        await original_batch_control.close()

    assert result == ('batch-1', 'attempt-1', manager_loop)


class BlockingDriver:
    stop_disposition = StopDisposition.IMMEDIATE

    def __init__(self) -> None:
        import asyncio

        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.stopped = asyncio.Event()

    async def start(self, handle, spec, config, callbacks):
        await callbacks.mark_running(endpoint_url=None, endpoint_ready=False, runtime_info={})
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    async def submit_command(self, handle, command_type, payload):
        raise AssertionError('not used')

    async def stop(self, handle):
        self.stopped.set()


class MarkRunningDriver:
    stop_disposition = StopDisposition.IMMEDIATE

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def start(self, handle, spec, config, callbacks):
        self.started.set()
        try:
            await callbacks.mark_running(endpoint_url=None, endpoint_ready=False, runtime_info={})
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    async def submit_command(self, handle, command_type, payload):
        raise AssertionError('not used')

    async def stop(self, handle):
        pass


class FailingCommandDriver(BlockingDriver):
    async def submit_command(self, handle, command_type, payload):
        raise RuntimeError('command failed')


class DeferredStopDriver(BlockingDriver):
    stop_disposition = StopDisposition.DEFERRED


class DriverWithoutStopDisposition:
    spec_type = 'missing_stop_disposition'

    async def start(self, handle, spec, config, callbacks):
        raise AssertionError('not used')

    async def submit_command(self, handle, command_type, payload):
        raise AssertionError('not used')

    async def stop(self, handle):
        raise AssertionError('not used')


class FailingStartExecutor:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.instance_mode = 'asyncio'

    async def start(self):
        pass

    async def close(self):
        pass

    async def has_assignment(self, assignment_id):
        return False

    async def capacity_snapshot(self):
        return ExecutorCapacitySnapshot(capacity_total=1, running_slot_cost=0, available_slots=1)

    async def start_assignment(self, **kwargs):
        self.started.set()
        raise RuntimeError('executor start failed')

    async def stop_assignment(self, assignment_id, timeout=10.0):
        pass

    async def submit_command(self, assignment_id, command_type, payload):
        pass

    async def collect_events_after(self, assignment_id, cursor, *, timeout=15.0):
        return []

    async def get_runtime_base_url(self, assignment_id, runtime_name):
        raise RuntimeError('not used')


class LoggingExecutor:
    instance_mode = 'asyncio'

    def __init__(self, *, capacity_total: int = 1) -> None:
        self.started = asyncio.Event()
        self.submitted = asyncio.Event()
        self.capacity_total = capacity_total
        self._assignments: dict[str, int] = {}

    async def start(self):
        pass

    async def close(self):
        pass

    async def has_assignment(self, assignment_id):
        return assignment_id in self._assignments

    async def capacity_snapshot(self):
        running_slot_cost = sum(self._assignments.values())
        return ExecutorCapacitySnapshot(
            capacity_total=self.capacity_total,
            running_slot_cost=running_slot_cost,
            available_slots=max(0, self.capacity_total - running_slot_cost),
        )

    async def start_assignment(self, assignment_id, execution_id, driver, spec, config, reporter, *, slot_cost=1):  # noqa: ARG002
        self._assignments[assignment_id] = slot_cost
        logging.getLogger('nexus.test.manager_context').info('executor start log')
        self.started.set()
        return ExecutionHandle(execution_id=execution_id, assignment_id=assignment_id)

    async def stop_assignment(self, assignment_id, timeout=10.0):  # noqa: ARG002
        self._assignments.pop(assignment_id, None)

    async def submit_command(self, assignment_id, command_type, payload):  # noqa: ARG002
        logging.getLogger('nexus.test.manager_context').info('executor command log')
        self.submitted.set()

    async def collect_events_after(self, assignment_id, cursor, *, timeout=15.0):  # noqa: ARG002
        return []

    async def get_runtime_base_url(self, assignment_id, runtime_name):  # noqa: ARG002
        raise RuntimeError('not used')

    async def get_assignment_info(self, assignment_id):  # noqa: ARG002
        return {}


def _flush_root_handlers() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()


def _json_log_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


async def test_instance_manager_reports_driver_construction_failure_before_executor_start(session_factory, monkeypatch):
    now = utcnow()
    driver = cast(ExecutionDriver, DriverWithoutStopDisposition())
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(aiomonitor_enabled=False),
        manager_id='im-driver-contract',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        executor=LoggingExecutor(),
        driver_factories={'nexus_single': lambda **_: driver},
    )
    execution = ExecutionRecord(
        execution_id='missing-stop-disposition',
        shard_id=1,
        mode=ExecutionMode.JOB,
        spec_type='nexus_single',
        instance_mode='asyncio',
        spec={'task_name': 'identity'},
        driver_config={},
        placement={},
        status=ExecutionStatus.ASSIGNED,
        route_status=RouteStatus.NONE,
        slot_cost=1,
        current_assignment_id='asn-missing-stop-disposition',
        current_attempt_id='attempt-1',
        attempt_count=0,
        max_attempts=1,
        next_schedule_at=now,
        created_at=now,
        updated_at=now,
    )
    assignment = AssignmentRecord(
        assignment_id='asn-missing-stop-disposition',
        execution_id=execution.execution_id,
        attempt_id='attempt-1',
        attempt_seq=1,
        manager_id=service.manager_id,
        manager_session_id=service.session_id,
        status=AssignmentStatus.PENDING,
        slot_cost=1,
        created_at=now,
        updated_at=now,
    )

    reported: dict = {}

    async def capture_start_failed(assignment_id, *, error):
        reported.update({'assignment_id': assignment_id, 'error': error})

    monkeypatch.setattr(service, '_mark_start_failed', capture_start_failed)
    try:
        assert not await service._start_assignment(execution, assignment)
        assert assignment.assignment_id not in service._running
        assert reported['assignment_id'] == assignment.assignment_id
        assert reported['error']['type'] == 'AssignmentStartFailed'
        assert reported['error']['cause_type'] == 'AttributeError'
        assert 'stop_disposition' in reported['error']['message']
    finally:
        await service.batch_control.close()


async def test_instance_manager_passes_persisted_parent_to_single_driver(session_factory):
    now = utcnow()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(aiomonitor_enabled=False),
        manager_id='im-batch-child',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={},
    )
    execution = ExecutionRecord(
        execution_id='exe_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        shard_id=1,
        mode=ExecutionMode.JOB,
        spec_type='nexus_single',
        instance_mode='asyncio',
        spec={
            'task_name': 'identity',
            'batch': {'parent_execution_id': 'exe_0123456789abcdef0123456789abcdef'},
        },
        driver_config={},
        placement={},
        status=ExecutionStatus.ASSIGNED,
        route_status=RouteStatus.NONE,
        slot_cost=1,
        current_assignment_id='asn-batch-child',
        current_attempt_id='attempt-1',
        attempt_count=0,
        max_attempts=1,
        next_schedule_at=now,
        created_at=now,
        updated_at=now,
        parent_execution_id='exe_0123456789abcdef0123456789abcdef',
    )
    assignment = AssignmentRecord(
        assignment_id='asn-batch-child',
        execution_id=execution.execution_id,
        attempt_id='attempt-1',
        attempt_seq=1,
        manager_id=service.manager_id,
        manager_session_id=service.session_id,
        status=AssignmentStatus.PENDING,
        slot_cost=1,
        created_at=now,
        updated_at=now,
    )

    try:
        driver = service._create_single_driver(
            execution=execution,
            assignment=assignment,
            internal_base_url='http://127.0.0.1:9001',
        )
        assert driver.parent_execution_id == execution.parent_execution_id
    finally:
        await service.batch_control.close()


async def test_instance_manager_resolves_batch_run_input_reference_in_memory(session_factory):
    from nexus.distributed.batch.task_store import BatchTaskRecord

    now = utcnow()
    parent_execution_id = 'exe_0123456789abcdef0123456789abcdef'
    child_execution_id = 'exe_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
    record = BatchTaskRecord.from_identity(
        parent_execution_id=parent_execution_id,
        identity={
            'version': 1,
            'item_id': 'item-1',
            'run_input': {'task_name': 'identity', 'task_config': {'marker': 'stored-only'}},
        },
    )

    class MemoryTaskStore:
        backend_name = 'tdsql'

        async def read_record(self, *, parent_execution_id, item_id):
            assert (parent_execution_id, item_id) == (record.parent_execution_id, record.item_id)
            return record

    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(aiomonitor_enabled=False),
        manager_id='im-task-store-child',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={},
        batch_task_store=MemoryTaskStore(),
    )
    execution = ExecutionRecord(
        execution_id=child_execution_id,
        shard_id=1,
        mode=ExecutionMode.JOB,
        spec_type='nexus_single',
        instance_mode='asyncio',
        spec={
            'spec_version': 1,
            'run_input_ref': {
                'version': 1,
                'store': 'tdsql',
                'parent_execution_id': parent_execution_id,
                'item_id': record.item_id,
                'record_sha256': record.record_sha256,
            },
            'batch': {'parent_execution_id': parent_execution_id, 'item_id': record.item_id, 'attempt_no': 0},
        },
        driver_config={},
        placement={},
        status=ExecutionStatus.ASSIGNED,
        route_status=RouteStatus.NONE,
        slot_cost=1,
        current_assignment_id='asn-task-store-child',
        current_attempt_id='attempt-1',
        attempt_count=0,
        max_attempts=1,
        next_schedule_at=now,
        created_at=now,
        updated_at=now,
        parent_execution_id=parent_execution_id,
    )

    try:
        resolved = await service._resolve_execution_spec(execution)
    finally:
        await service.batch_control.close()

    assert resolved.temporary_task_files is None
    assert 'run_input_ref' not in resolved.spec
    assert resolved.spec['run_input']['task_name'] == 'identity'
    assert resolved.spec['run_input']['task_config'] == {'marker': 'stored-only'}
    assert resolved.spec['run_input']['run_id'] == child_execution_id
    assert 'run_input' not in execution.spec


async def test_instance_manager_v3_records_materialize_isolated_private_local_zips(session_factory, tmp_path):
    parent_execution_id = 'exe_v3_parent'
    records = {
        'item-a': _v3_task_record(
            tmp_path,
            parent_execution_id=parent_execution_id,
            item_id='item-a',
            content='payload-a',
        ),
        'item-b': _v3_task_record(
            tmp_path,
            parent_execution_id=parent_execution_id,
            item_id='item-b',
            content='payload-b',
        ),
    }

    class MemoryTaskStore:
        backend_name = 'tdsql'

        async def read_record(self, *, parent_execution_id, item_id):
            assert parent_execution_id == 'exe_v3_parent'
            return records[item_id]

        async def close(self):
            return None

    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(
            aiomonitor_enabled=False,
            batch_control_spool_dir=str(tmp_path / 'spool'),
        ),
        manager_id='im-v3-task-files',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=2,
        driver_factories={},
        batch_task_store=MemoryTaskStore(),
    )
    resolved_items = []
    try:
        for suffix in ('a', 'b'):
            assignment_id = f'asn-v3-{suffix}'
            execution = _task_store_child_execution(
                parent_execution_id=parent_execution_id,
                child_execution_id=f'exe-v3-{suffix}',
                assignment_id=assignment_id,
                record=records[f'item-{suffix}'],
            )
            resolved = await service._resolve_execution_spec(execution, assignment_id=assignment_id)
            resolved_items.append(resolved)

            assert resolved.temporary_task_files is not None
            task_files_spec = resolved.spec['run_input']['task_files']
            archive_path = Path(task_files_spec['path'])
            assignment_root = archive_path.parent
            assert assignment_root.parent == service._assignment_files_root
            assert list(assignment_root.iterdir()) == [archive_path]
            assert stat.S_IMODE(assignment_root.stat().st_mode) == 0o700
            assert stat.S_IMODE(archive_path.stat().st_mode) == 0o600
            assert task_files_spec['type'] == 'zip'
            assert task_files_spec['size'] == archive_path.stat().st_size
            assert resolved.spec['run_input']['run_id'] == execution.execution_id

            files = task_files_from_spec(task_files_spec, run_id=execution.execution_id)
            assert isinstance(files, LocalZipTaskFiles)
            assert await files.read_text('input.txt') == f'payload-{suffix}'
            assert 'run_input' not in execution.spec

        assert stat.S_IMODE(service._assignment_files_root.stat().st_mode) == 0o700
        assert resolved_items[0].temporary_task_files.directory != resolved_items[1].temporary_task_files.directory
    finally:
        for resolved in resolved_items:
            if resolved.temporary_task_files is not None:
                resolved.temporary_task_files.cleanup()
        await service.stop()


@pytest.mark.parametrize('invalid_kind', ['nxbr', 'frame', 'zip', 'record_limit', 'frame_limit'])
async def test_instance_manager_rejects_invalid_v3_task_files_before_executor_start(
    session_factory,
    tmp_path,
    monkeypatch,
    invalid_kind,
):
    parent_execution_id = 'exe-v3-invalid-parent'
    assignment_id = f'asn-v3-invalid-{invalid_kind}'
    item_id = f'item-invalid-{invalid_kind}'
    if invalid_kind in {'nxbr', 'frame', 'zip'}:
        record = _malformed_v3_task_record(
            tmp_path,
            parent_execution_id=parent_execution_id,
            item_id=item_id,
            mutation=invalid_kind,
        )
    else:
        record = _v3_task_record(
            tmp_path,
            parent_execution_id=parent_execution_id,
            item_id=item_id,
            content='hard limit payload',
        )

    config_values = {
        'aiomonitor_enabled': False,
        'batch_control_spool_dir': str(tmp_path / f'spool-invalid-{invalid_kind}'),
    }
    if invalid_kind == 'record_limit':
        config_values['batch_task_record_max_size_bytes'] = len(record.record_data) - 1
    elif invalid_kind == 'frame_limit':
        frame_size = struct.unpack_from('<4sHHQQ', record.record_data)[4]
        config_values['batch_data_frame_max_size_bytes'] = frame_size - 1

    class MemoryTaskStore:
        backend_name = 'tdsql'

        async def read_record(self, **_kwargs):
            return record

        async def close(self):
            return None

    executor = LoggingExecutor()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(**config_values),
        manager_id=f'im-v3-invalid-{invalid_kind}',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        executor=executor,
        driver_factories={'nexus_single': lambda **_: CompletingDriver()},
        batch_task_store=MemoryTaskStore(),
    )
    execution = _task_store_child_execution(
        parent_execution_id=parent_execution_id,
        child_execution_id=f'exe-v3-invalid-{invalid_kind}',
        assignment_id=assignment_id,
        record=record,
    )
    now = utcnow()
    assignment = AssignmentRecord(
        assignment_id=assignment_id,
        execution_id=execution.execution_id,
        attempt_id='attempt-1',
        attempt_seq=1,
        manager_id=service.manager_id,
        manager_session_id=service.session_id,
        status=AssignmentStatus.PENDING,
        slot_cost=1,
        created_at=now,
        updated_at=now,
    )
    mark_start_failed = AsyncMock()
    monkeypatch.setattr(service, '_mark_start_failed', mark_start_failed)
    try:
        assert not await service._start_assignment(execution, assignment)
        assert not executor.started.is_set()
        assert assignment_id not in service._running
        assert list(service._assignment_files_root.iterdir()) == []
        mark_start_failed.assert_awaited_once()
    finally:
        await service.stop()


@pytest.mark.parametrize('failure_kind', ['error', 'timeout_absent'])
async def test_instance_manager_cleans_v3_task_files_when_executor_start_is_confirmed_failed(
    session_factory,
    tmp_path,
    monkeypatch,
    failure_kind,
):
    parent_execution_id = 'exe-v3-start-parent'
    assignment_id = f'asn-v3-start-{failure_kind}'
    record = _v3_task_record(
        tmp_path,
        parent_execution_id=parent_execution_id,
        item_id=f'item-{failure_kind}',
        content='start failure payload',
    )

    class MemoryTaskStore:
        backend_name = 'tdsql'

        async def read_record(self, **_kwargs):
            return record

        async def close(self):
            return None

    class StartFailureExecutor(LoggingExecutor):
        async def start_assignment(self, **_kwargs):
            self.started.set()
            if failure_kind == 'timeout_absent':
                raise TimeoutError('executor start timed out')
            raise RuntimeError('executor start failed')

    executor = StartFailureExecutor()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(
            aiomonitor_enabled=False,
            batch_control_spool_dir=str(tmp_path / f'spool-{failure_kind}'),
        ),
        manager_id=f'im-v3-start-{failure_kind}',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        executor=executor,
        driver_factories={'nexus_single': lambda **_: CompletingDriver()},
        batch_task_store=MemoryTaskStore(),
    )
    execution = _task_store_child_execution(
        parent_execution_id=parent_execution_id,
        child_execution_id=f'exe-v3-start-{failure_kind}',
        assignment_id=assignment_id,
        record=record,
    )
    now = utcnow()
    assignment = AssignmentRecord(
        assignment_id=assignment_id,
        execution_id=execution.execution_id,
        attempt_id='attempt-1',
        attempt_seq=1,
        manager_id=service.manager_id,
        manager_session_id=service.session_id,
        status=AssignmentStatus.PENDING,
        slot_cost=1,
        created_at=now,
        updated_at=now,
    )
    mark_start_failed = AsyncMock()
    monkeypatch.setattr(service, '_mark_start_failed', mark_start_failed)
    try:
        assert not await service._start_assignment(execution, assignment)
        assert assignment_id not in service._running
        assert not (service._assignment_files_root / assignment_id).exists()
        mark_start_failed.assert_awaited_once()
    finally:
        await service.stop()


@pytest.mark.parametrize('executor_state', ['present', 'unknown'])
async def test_instance_manager_retains_v3_task_files_when_timed_out_start_may_exist(
    session_factory,
    tmp_path,
    monkeypatch,
    executor_state,
):
    parent_execution_id = 'exe-v3-timeout-parent'
    assignment_id = f'asn-v3-timeout-{executor_state}'
    record = _v3_task_record(
        tmp_path,
        parent_execution_id=parent_execution_id,
        item_id=f'item-timeout-{executor_state}',
        content=f'timeout {executor_state} payload',
    )

    class MemoryTaskStore:
        backend_name = 'tdsql'

        async def read_record(self, **_kwargs):
            return record

        async def close(self):
            return None

    class StartTimeoutExecutor(LoggingExecutor):
        async def start_assignment(self, **_kwargs):
            self.started.set()
            raise TimeoutError('executor start timed out')

        async def has_assignment(self, _assignment_id):
            if executor_state == 'unknown':
                raise TimeoutError('executor assignment state unavailable')
            return True

    executor = StartTimeoutExecutor()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(
            aiomonitor_enabled=False,
            batch_control_spool_dir=str(tmp_path / f'spool-timeout-{executor_state}'),
        ),
        manager_id=f'im-v3-timeout-{executor_state}',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        executor=executor,
        driver_factories={'nexus_single': lambda **_: CompletingDriver()},
        batch_task_store=MemoryTaskStore(),
    )
    execution = _task_store_child_execution(
        parent_execution_id=parent_execution_id,
        child_execution_id=f'exe-v3-timeout-{executor_state}',
        assignment_id=assignment_id,
        record=record,
    )
    now = utcnow()
    assignment = AssignmentRecord(
        assignment_id=assignment_id,
        execution_id=execution.execution_id,
        attempt_id='attempt-1',
        attempt_seq=1,
        manager_id=service.manager_id,
        manager_session_id=service.session_id,
        status=AssignmentStatus.PENDING,
        slot_cost=1,
        created_at=now,
        updated_at=now,
    )
    mark_start_failed = AsyncMock()
    monkeypatch.setattr(service, '_mark_start_failed', mark_start_failed)
    try:
        assert await service._start_assignment(execution, assignment)
        assert executor.started.is_set()
        running = service._running[assignment_id]
        assert running.temporary_task_files is not None
        assert running.temporary_task_files.zip_path.exists()
        mark_start_failed.assert_not_awaited()
        if executor_state == 'unknown':
            assert not service._executor_health.responsive
    finally:
        await service.stop()


async def test_instance_manager_cleans_v3_task_files_if_running_registration_fails(
    session_factory,
    tmp_path,
    monkeypatch,
):
    parent_execution_id = 'exe-v3-register-parent'
    assignment_id = 'asn-v3-register-failure'
    record = _v3_task_record(
        tmp_path,
        parent_execution_id=parent_execution_id,
        item_id='item-register-failure',
        content='registration failure payload',
    )

    class MemoryTaskStore:
        backend_name = 'tdsql'

        async def read_record(self, **_kwargs):
            return record

        async def close(self):
            return None

    class RejectingRunningRegistry(dict):
        def __setitem__(self, key, value):  # noqa: ARG002
            raise RuntimeError('running assignment registration failed')

    executor = LoggingExecutor()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(
            aiomonitor_enabled=False,
            batch_control_spool_dir=str(tmp_path / 'spool-register-failure'),
        ),
        manager_id='im-v3-register-failure',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        executor=executor,
        driver_factories={'nexus_single': lambda **_: CompletingDriver()},
        batch_task_store=MemoryTaskStore(),
    )
    execution = _task_store_child_execution(
        parent_execution_id=parent_execution_id,
        child_execution_id='exe-v3-register-failure',
        assignment_id=assignment_id,
        record=record,
    )
    now = utcnow()
    assignment = AssignmentRecord(
        assignment_id=assignment_id,
        execution_id=execution.execution_id,
        attempt_id='attempt-1',
        attempt_seq=1,
        manager_id=service.manager_id,
        manager_session_id=service.session_id,
        status=AssignmentStatus.PENDING,
        slot_cost=1,
        created_at=now,
        updated_at=now,
    )
    monkeypatch.setattr(service, '_running', RejectingRunningRegistry())
    monkeypatch.setattr(service, '_mark_start_failed', AsyncMock())
    try:
        assert not await service._start_assignment(execution, assignment)
        assert not executor.started.is_set()
        assert not (service._assignment_files_root / assignment_id).exists()
    finally:
        await service.stop()


@pytest.mark.parametrize('terminal_kind', ['complete', 'fail', 'killed'])
async def test_instance_manager_terminal_paths_clean_assignment_task_files(
    session_factory,
    tmp_path,
    monkeypatch,
    terminal_kind,
):
    assignment_id = f'asn-v3-terminal-{terminal_kind}'
    temporary = _temporary_task_files(tmp_path, assignment_id)
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(
            aiomonitor_enabled=False,
            batch_control_spool_dir=str(tmp_path / f'spool-terminal-{terminal_kind}'),
        ),
        manager_id=f'im-v3-terminal-{terminal_kind}',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={},
    )
    service._running[assignment_id] = RunningAssignment(
        execution_id=f'exe-v3-terminal-{terminal_kind}',
        assignment_id=assignment_id,
        slot_cost=1,
        stop_timeout_seconds=1,
        stop_disposition=StopDisposition.IMMEDIATE,
        temporary_task_files=temporary,
    )
    repository_method = {
        'complete': 'complete_execution',
        'fail': 'fail_execution',
        'killed': 'mark_killed',
    }[terminal_kind]
    monkeypatch.setattr(ManagerRepository, repository_method, AsyncMock())
    try:
        if terminal_kind == 'complete':
            await service._complete(assignment_id, result={'ok': True})
        elif terminal_kind == 'fail':
            await service._fail(assignment_id, error={'type': 'TaskFailed'})
        else:
            await service._mark_killed(assignment_id)

        assert assignment_id not in service._running
        assert not temporary.directory.exists()
    finally:
        await service.stop()


async def test_instance_manager_cleanup_failure_does_not_replace_terminal_result(
    session_factory,
    tmp_path,
    monkeypatch,
    caplog,
):
    assignment_id = 'asn-v3-cleanup-error'
    temporary = _temporary_task_files(tmp_path, assignment_id)
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(
            aiomonitor_enabled=False,
            batch_control_spool_dir=str(tmp_path / 'spool-cleanup-error'),
        ),
        manager_id='im-v3-cleanup-error',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={},
    )
    service._running[assignment_id] = RunningAssignment(
        execution_id='exe-v3-cleanup-error',
        assignment_id=assignment_id,
        slot_cost=1,
        stop_timeout_seconds=1,
        stop_disposition=StopDisposition.IMMEDIATE,
        temporary_task_files=temporary,
    )
    monkeypatch.setattr(ManagerRepository, 'complete_execution', AsyncMock())

    def fail_cleanup(_temporary):
        raise OSError('cleanup failed')

    monkeypatch.setattr(TemporaryTaskFiles, 'cleanup', fail_cleanup)

    with caplog.at_level(logging.WARNING):
        await service._complete(assignment_id, result={'ok': True})

    assert assignment_id not in service._running
    assert 'Failed to clean assignment task files' in caplog.text
    await service.stop()


async def test_instance_manager_stop_cleans_running_task_files_and_manager_root(session_factory, tmp_path):
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(
            aiomonitor_enabled=False,
            batch_control_spool_dir=str(tmp_path / 'spool-manager-stop'),
        ),
        manager_id='im-v3-manager-stop',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        executor=LoggingExecutor(),
        driver_factories={},
    )
    assignment_id = 'asn-v3-manager-stop'
    directory = service._assignment_files_root / assignment_id
    directory.mkdir(mode=0o700)
    archive_path = directory / 'files.zip'
    archive_path.write_bytes(b'test zip placeholder')
    temporary = TemporaryTaskFiles(directory=directory, zip_path=archive_path)
    service._running[assignment_id] = RunningAssignment(
        execution_id='exe-v3-manager-stop',
        assignment_id=assignment_id,
        slot_cost=1,
        stop_timeout_seconds=1,
        stop_disposition=StopDisposition.IMMEDIATE,
        temporary_task_files=temporary,
    )
    manager_root = service._assignment_files_root

    await service.stop()

    assert not temporary.directory.exists()
    assert not manager_root.exists()


@pytest.mark.parametrize('failure_kind', ['stop_assignment', 'executor_close'])
async def test_instance_manager_stop_cleans_manager_root_when_executor_shutdown_fails(
    session_factory,
    tmp_path,
    failure_kind,
):
    class FailingShutdownExecutor(LoggingExecutor):
        async def stop_assignment(self, assignment_id, timeout=10.0):
            if failure_kind == 'stop_assignment':
                raise RuntimeError('executor assignment stop failed')
            await super().stop_assignment(assignment_id, timeout=timeout)

        async def close(self):
            if failure_kind == 'executor_close':
                raise RuntimeError('executor close failed')

    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(
            aiomonitor_enabled=False,
            batch_control_spool_dir=str(tmp_path / f'spool-stop-failure-{failure_kind}'),
        ),
        manager_id=f'im-v3-stop-failure-{failure_kind}',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        executor=FailingShutdownExecutor(),
        driver_factories={},
    )
    assignment_id = f'asn-v3-stop-failure-{failure_kind}'
    directory = service._assignment_files_root / assignment_id
    directory.mkdir(mode=0o700)
    archive_path = directory / 'files.zip'
    archive_path.write_bytes(b'test zip placeholder')
    temporary = TemporaryTaskFiles(directory=directory, zip_path=archive_path)
    service._running[assignment_id] = RunningAssignment(
        execution_id=f'exe-v3-stop-failure-{failure_kind}',
        assignment_id=assignment_id,
        slot_cost=1,
        stop_timeout_seconds=1,
        stop_disposition=StopDisposition.IMMEDIATE,
        temporary_task_files=temporary,
    )
    manager_root = service._assignment_files_root

    with pytest.raises(RuntimeError, match='executor (assignment stop|close) failed'):
        await service.stop()

    assert not temporary.directory.exists()
    assert not manager_root.exists()


@pytest.mark.parametrize('release_kind', ['killed_if_active', 'deferred_stop'])
@pytest.mark.parametrize('released', [True, False])
async def test_instance_manager_conditional_stop_paths_clean_only_after_database_release(
    session_factory,
    tmp_path,
    monkeypatch,
    release_kind,
    released,
):
    assignment_id = f'asn-v3-{release_kind}-{released}'
    temporary = _temporary_task_files(tmp_path, assignment_id)
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(
            aiomonitor_enabled=False,
            batch_control_spool_dir=str(tmp_path / f'spool-{release_kind}-{released}'),
        ),
        manager_id=f'im-v3-{release_kind}-{released}',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={},
    )
    service._running[assignment_id] = RunningAssignment(
        execution_id=f'exe-v3-{release_kind}-{released}',
        assignment_id=assignment_id,
        slot_cost=1,
        stop_timeout_seconds=1,
        stop_disposition=StopDisposition.IMMEDIATE,
        temporary_task_files=temporary,
    )
    repository_method = (
        'mark_killed_if_active' if release_kind == 'killed_if_active' else 'release_deferred_stop_if_active'
    )
    monkeypatch.setattr(ManagerRepository, repository_method, AsyncMock(return_value=released))
    try:
        if release_kind == 'killed_if_active':
            await service._mark_killed_if_active(assignment_id)
        else:
            await service._release_deferred_stop_if_active(assignment_id)

        assert (assignment_id not in service._running) is released
        assert temporary.directory.exists() is not released
    finally:
        await service._release_running_assignment(assignment_id)
        await service.stop()


async def test_instance_manager_live_driver_uses_default_local_trace_registry(session_factory, tmp_path):
    from nexus.distributed.execution.trace_backends import RunTraceSinkRequest

    now = utcnow()
    config = DistributedConfig(
        shard_count=8,
        manager_healthy_ttl=timedelta(seconds=15),
        run_trace_artifact_root=str(tmp_path),
    )
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='im-live',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={},
    )
    execution = ExecutionRecord(
        execution_id='exec-live',
        shard_id=1,
        mode=ExecutionMode.LIVE,
        spec_type='nexus_live',
        instance_mode='asyncio',
        spec={'task_name': 'identity'},
        driver_config={},
        placement={},
        status=ExecutionStatus.ASSIGNED,
        route_status=RouteStatus.ACTIVE,
        slot_cost=1,
        current_assignment_id='asn-live',
        current_attempt_id='attempt-1',
        attempt_count=0,
        max_attempts=1,
        owner_principal_id='principal-owner-a',
        next_schedule_at=now,
        created_at=now,
        updated_at=now,
    )
    assignment = AssignmentRecord(
        assignment_id='asn-live',
        execution_id='exec-live',
        attempt_id='attempt-1',
        attempt_seq=1,
        manager_id='im-live',
        manager_session_id=service.session_id,
        status=AssignmentStatus.PENDING,
        slot_cost=1,
        created_at=now,
        updated_at=now,
    )

    driver = service._create_live_driver(
        execution=execution,
        assignment=assignment,
        internal_base_url='http://127.0.0.1:9001/v1/internal/executions/asn-live',
    )

    resolved = driver.trace_registry.resolve(
        run_kind='live',
        driver_config={},
        request=RunTraceSinkRequest(execution_id='exec-live', assignment_id='asn-live', attempt_id='attempt-1'),
    )
    assert str(resolved.backend_sinks['local'].root) == str(tmp_path)
    assert driver.attempt_id == 'attempt-1'
    assert driver.owner_principal_id == 'principal-owner-a'


async def test_instance_manager_assignment_logs_include_distributed_execution_context(session_factory, tmp_path):
    distributed_logs.setup_distributed_logging(
        log_dir=str(tmp_path),
        log_level='INFO',
        component='instance-manager',
        identity='im-context',
    )
    executor = LoggingExecutor()
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='im-context',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        executor=executor,
        driver_factories={'nexus_single': lambda **_: CompletingDriver()},
    )
    await service.start()
    try:
        async with session_factory() as session:
            execution_repo = ExecutionRepository(session, shard_count=8)
            execution = await execution_repo.create_execution(
                CreateExecution(
                    execution_id='job-log-context',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={'task_name': 'identity'},
                ),
                now=utcnow(),
            )
            scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
            assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
            await session.commit()

        assert assignment is not None
        await service.poll_assignments_once()
        await asyncio.wait_for(executor.started.wait(), timeout=2)
        _flush_root_handlers()

        payloads = _json_log_lines(tmp_path / 'instance-manager-im-context.jsonl')
        payload = next(item for item in payloads if item['message'] == 'executor start log')
        assert payload['execution_id'] == 'job-log-context'
        assert payload['assignment_id'] == assignment.assignment_id
        assert payload['attempt_id'] == assignment.attempt_id
        assert payload['manager_id'] == 'im-context'
        assert payload['manager_session_id'] == service.session_id
    finally:
        await service.stop()


async def test_instance_manager_starts_assignment_under_persisted_carrier(session_factory, monkeypatch):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from nexus.utils.otel.propagation import inject_carrier

    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    provider = TracerProvider(shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, 'get_tracer', provider.get_tracer)
    tracer = provider.get_tracer('tests.distributed.manager')
    executor = LoggingExecutor()
    observed_current_span_ids: list[int] = []
    original_start_assignment = executor.start_assignment

    async def capture_start_assignment(*args, **kwargs):
        observed_current_span_ids.append(trace.get_current_span().get_span_context().span_id)
        return await original_start_assignment(*args, **kwargs)

    monkeypatch.setattr(executor, 'start_assignment', capture_start_assignment)
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15)),
        manager_id='im-trace-carrier',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        executor=executor,
        driver_factories={'nexus_single': lambda **_: CompletingDriver()},
    )
    await service.start()
    try:
        with tracer.start_as_current_span('scheduler.assign') as scheduler_span:
            assignment_carrier = inject_carrier()
        async with session_factory() as session:
            execution_repo = ExecutionRepository(session, shard_count=8)
            execution = await execution_repo.create_execution(
                CreateExecution(
                    execution_id='manager-consumes-trace-carrier',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={'task_name': 'identity'},
                ),
                now=utcnow(),
            )
            scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
            assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
            assert assignment is not None
            row = await session.get(AssignmentRow, assignment.assignment_id)
            assert row is not None
            row.trace_carrier_json = assignment_carrier
            await session.commit()

        assert await service.poll_assignments_once() == 1
        consumer_span = next(
            span for span in exporter.get_finished_spans() if span.name == 'distributed.assignment.start'
        )
        assert consumer_span.parent is not None
        assert consumer_span.parent.span_id == scheduler_span.get_span_context().span_id
        assert observed_current_span_ids == [consumer_span.context.span_id]
    finally:
        await service.stop()
        provider.shutdown()


async def test_instance_manager_serializes_poll_and_command_work(session_factory):
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='im-serial',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={},
    )
    await service.start()
    poll_running = 0
    poll_max = 0
    command_running = 0
    command_max = 0

    async def fake_poll_unlocked():
        nonlocal poll_running, poll_max
        poll_running += 1
        poll_max = max(poll_max, poll_running)
        await asyncio.sleep(0.01)
        poll_running -= 1
        return 1

    async def fake_command_unlocked():
        nonlocal command_running, command_max
        command_running += 1
        command_max = max(command_max, command_running)
        await asyncio.sleep(0.01)
        command_running -= 1
        return 1

    service._poll_assignments_once_unlocked = fake_poll_unlocked
    service._claim_commands_once_unlocked = fake_command_unlocked
    try:
        assert await asyncio.gather(service.poll_assignments_once(), service.poll_assignments_once()) == [1, 1]
        assert await asyncio.gather(service.claim_commands_once(), service.claim_commands_once()) == [1, 1]
    finally:
        await service.stop()

    assert poll_max == 1
    assert command_max == 1


async def test_instance_manager_heartbeat_loop_self_fences_when_heartbeat_hangs(session_factory):
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(
            shard_count=8,
            manager_heartbeat_interval=0.005,
            manager_healthy_ttl=timedelta(seconds=0.01),
            manager_self_fence_timeout=timedelta(seconds=0.03),
            manager_dead_ttl=timedelta(seconds=0.06),
        ),
        manager_id='im-hung-heartbeat',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={},
    )
    blocked = asyncio.Event()
    reasons: list[str] = []
    original_self_fence = service._self_fence

    async def hang_heartbeat() -> bool:
        await blocked.wait()
        return True

    async def capture_self_fence(reason: str) -> None:
        reasons.append(reason)
        await original_self_fence(reason)

    service.heartbeat_once = hang_heartbeat
    service._self_fence = capture_self_fence

    await asyncio.wait_for(service._heartbeat_loop(), timeout=0.2)

    assert service._stopping.is_set()
    assert reasons == ['heartbeat_timeout']


@pytest.mark.parametrize('error_kind', [1053, 1158, 1159, 1160, 1161, 2003, 2013, 'pool_timeout'])
async def test_instance_manager_heartbeat_loop_retries_transient_database_errors_until_self_fence(
    session_factory, error_kind
):
    class DatabaseError(Exception):
        pass

    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(
            shard_count=8,
            manager_heartbeat_interval=0.005,
            manager_healthy_ttl=timedelta(seconds=0.01),
            manager_self_fence_timeout=timedelta(seconds=0.03),
            manager_dead_ttl=timedelta(seconds=0.06),
        ),
        manager_id='im-db-heartbeat-timeout',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={},
    )
    attempts = 0
    reasons: list[str] = []
    original_self_fence = service._self_fence

    async def fail_heartbeat() -> bool:
        nonlocal attempts
        attempts += 1
        if error_kind == 'pool_timeout':
            raise SqlAlchemyTimeoutError('QueuePool limit reached')
        raise OperationalError('heartbeat', {}, DatabaseError(error_kind, 'connection failed or lost'))

    async def capture_self_fence(reason: str) -> None:
        reasons.append(reason)
        await original_self_fence(reason)

    service.heartbeat_once = fail_heartbeat
    service._self_fence = capture_self_fence

    await asyncio.wait_for(service._heartbeat_loop(), timeout=0.2)

    assert attempts > 1
    assert service._stopping.is_set()
    assert reasons == ['heartbeat_timeout']


@pytest.mark.parametrize('errno', [1053, 1158, 1159, 1160, 1161, 2003])
async def test_instance_manager_heartbeat_recovers_after_transient_mysql_error(session_factory, errno):
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(),
        manager_id='im-mysql-recovery',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={},
    )
    heartbeat = service.heartbeat_once
    attempts = 0
    recovered = False

    async def flaky_heartbeat():
        nonlocal attempts, recovered
        attempts += 1
        if attempts == 1:
            raise OperationalError('heartbeat', {}, Exception(errno, 'temporary database failure'))
        recovered = await heartbeat()
        service._stopping.set()
        return recovered

    try:
        assert await service._register_session()
        service.heartbeat_once = flaky_heartbeat
        service._wait_for_stop = AsyncMock()
        service._self_fence = AsyncMock()

        await asyncio.wait_for(service._heartbeat_loop(), timeout=1)

        assert attempts == 2
        assert recovered is True
        service._self_fence.assert_not_awaited()
        async with session_factory() as session:
            liveness = await session.get(InstanceManagerLivenessRow, service.manager_id)
            assert liveness is not None
            assert liveness.fenced_at is None
            assert liveness.session_id == service.session_id
    finally:
        await service.stop()


async def test_instance_manager_heartbeat_recovers_after_pool_connection_is_returned(tmp_path, caplog):
    engine = create_async_engine(
        f'sqlite+aiosqlite:///{tmp_path / "heartbeat.db"}',
        pool_size=2,
        max_overflow=0,
        pool_timeout=0.01,
    )
    await create_schema(engine)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    service = InstanceManagerService(
        session_factory=factory,
        config=DistributedConfig(
            manager_heartbeat_interval=0.005,
            manager_healthy_ttl=timedelta(seconds=0.01),
            manager_self_fence_timeout=timedelta(seconds=1),
            manager_dead_ttl=timedelta(seconds=2),
        ),
        manager_id='im-pool-recovery',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={},
    )
    held_connections = []
    heartbeat = service.heartbeat_once
    wait_for_stop = service._wait_for_stop
    reasons = []
    recovered = False

    async def stop_after_success():
        nonlocal recovered
        recovered = await heartbeat()
        service._stopping.set()
        return recovered

    async def release_connection_during_backoff(delay):
        if held_connections:
            await held_connections.pop().close()
        await wait_for_stop(delay)

    async def capture_self_fence(reason):
        reasons.append(reason)
        service._stopping.set()

    try:
        assert await service._register_session()
        held_connections.extend([await engine.connect(), await engine.connect()])
        service.heartbeat_once = stop_after_success
        service._wait_for_stop = release_connection_during_backoff
        service._self_fence = capture_self_fence

        await asyncio.wait_for(service._heartbeat_loop(), timeout=2)

        assert recovered is True
        assert reasons == []
        retry_logs = [
            record for record in caplog.records if 'Retrying Instance manager heartbeat' in record.getMessage()
        ]
        assert len(retry_logs) == 1
        assert retry_logs[0].exc_info[0] is SqlAlchemyTimeoutError
        async with factory() as session:
            liveness = await session.get(InstanceManagerLivenessRow, service.manager_id)
            assert liveness is not None
            assert liveness.fenced_at is None
            assert liveness.session_id == service.session_id
    finally:
        for connection in held_connections:
            await connection.close()
        await service.stop()
        await engine.dispose()


@pytest.mark.parametrize(
    ('error', 'expected_reason'),
    [
        (OperationalError('heartbeat', {}, Exception(1142, 'permission denied')), 'heartbeat_fatal_error'),
        (OperationalError('heartbeat', {}, Exception(1040, 'too many connections')), 'heartbeat_fatal_error'),
        (OperationalError('heartbeat', {}, Exception(1203, 'too many user connections')), 'heartbeat_fatal_error'),
        (RuntimeError('unexpected heartbeat failure'), 'heartbeat_fatal_error'),
        (asyncio.CancelledError(), None),
    ],
    ids=['permanent_database_error', 'connection_limit', 'user_connection_limit', 'unexpected_error', 'cancelled'],
)
async def test_instance_manager_heartbeat_does_not_retry_fatal_errors_or_cancellation(
    session_factory, error, expected_reason
):
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(),
        manager_id='im-fatal-heartbeat',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={},
    )
    service.heartbeat_once = AsyncMock(side_effect=error)
    service._wait_for_stop = AsyncMock()
    service._self_fence = AsyncMock()

    try:
        if expected_reason is None:
            with pytest.raises(asyncio.CancelledError):
                await service._heartbeat_loop()
            service._self_fence.assert_not_awaited()
        else:
            await service._heartbeat_loop()
            service._self_fence.assert_awaited_once_with(expected_reason)
        service.heartbeat_once.assert_awaited_once()
        service._wait_for_stop.assert_not_awaited()
    finally:
        await service.stop()


async def test_instance_manager_marks_stop_command_failed_when_killed_update_fails(session_factory):
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    driver = BlockingDriver()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='im-1',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={'nexus_live': lambda **_: driver},
    )
    await service.start()

    async def fail_mark_killed(_assignment_id):
        raise RuntimeError('db update failed')

    service._mark_killed_if_active = fail_mark_killed
    try:
        async with session_factory() as session:
            execution_repo = ExecutionRepository(session, shard_count=8)
            execution = await execution_repo.create_execution(
                CreateExecution(
                    execution_id='live-stop-killed-update-fails',
                    mode=ExecutionMode.LIVE,
                    spec_type='nexus_live',
                    instance_mode='asyncio',
                    spec={'task_name': 'identity'},
                    route_status=RouteStatus.ACTIVE,
                    stop_timeout_seconds=3,
                ),
                now=utcnow(),
            )
            scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
            assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
            await session.commit()

        assert assignment is not None
        await service.poll_assignments_once()
        await asyncio.wait_for(driver.started.wait(), timeout=2)

        async with session_factory() as session:
            repo = ExecutionRepository(session, shard_count=8)
            await repo.request_stop('live-stop-killed-update-fails', now=utcnow())
            await session.commit()

        assert await service.claim_commands_once() == 1

        async with session_factory() as session:
            command = await session.scalar(
                select(ExecutionCommandRow).where(
                    ExecutionCommandRow.execution_id == 'live-stop-killed-update-fails',
                    ExecutionCommandRow.type == 'stop.request',
                )
            )

        assert command is not None
        assert command.status == CommandStatus.FAILED
        assert command.error_json['type'] == 'RuntimeError'
    finally:
        await service.stop()


async def test_instance_manager_deferred_stop_releases_assignment_without_terminalizing_parent(session_factory):
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    driver = DeferredStopDriver()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='batch-manager',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        workload='orchestration',
        driver_factories={'nexus_batch': lambda **_: driver},
    )
    await service.start()
    try:
        async with session_factory() as session:
            execution_repo = ExecutionRepository(session, shard_count=8)
            execution = await execution_repo.create_execution(
                CreateExecution(
                    execution_id='batch-deferred-stop',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_batch',
                    instance_mode='asyncio',
                    spec={},
                    route_status=RouteStatus.NONE,
                    stop_timeout_seconds=3,
                ),
                now=utcnow(),
            )
            scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
            assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
            await session.commit()

        assert assignment is not None
        await service.poll_assignments_once()
        await asyncio.wait_for(driver.started.wait(), timeout=2)

        async with session_factory() as session:
            await ExecutionRepository(session, shard_count=8).request_stop('batch-deferred-stop', now=utcnow())
            await session.commit()

        assert await service.claim_commands_once() == 1

        async with session_factory() as session:
            execution_row = await session.get(ExecutionRow, 'batch-deferred-stop')
            assignment_row = await session.get(AssignmentRow, assignment.assignment_id)
            command = await session.scalar(
                select(ExecutionCommandRow).where(
                    ExecutionCommandRow.execution_id == 'batch-deferred-stop',
                    ExecutionCommandRow.type == 'stop.request',
                )
            )

        assert execution_row is not None
        assert execution_row.status == ExecutionStatus.STOPPING
        assert assignment_row is not None
        assert assignment_row.status == AssignmentStatus.RELEASED
        assert command is not None
        assert command.status == CommandStatus.DONE
    finally:
        await service.stop()


async def test_instance_manager_mark_killed_if_active_ignores_finished_assignment(session_factory):
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    driver = BlockingDriver()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='im-1',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={'nexus_live': lambda **_: driver},
    )
    await service.start()
    try:
        async with session_factory() as session:
            execution_repo = ExecutionRepository(session, shard_count=8)
            execution = await execution_repo.create_execution(
                CreateExecution(
                    execution_id='live-stop-already-finished',
                    mode=ExecutionMode.LIVE,
                    spec_type='nexus_live',
                    instance_mode='asyncio',
                    spec={'task_name': 'identity'},
                    route_status=RouteStatus.ACTIVE,
                    stop_timeout_seconds=3,
                ),
                now=utcnow(),
            )
            scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
            assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
            await session.commit()

        assert assignment is not None
        await service.poll_assignments_once()
        await asyncio.wait_for(driver.started.wait(), timeout=2)

        await service._mark_killed(assignment.assignment_id)
        await service._mark_killed_if_active(assignment.assignment_id)

        async with session_factory() as session:
            row = await session.scalar(
                select(AssignmentRow).where(AssignmentRow.assignment_id == assignment.assignment_id)
            )

        assert row is not None
        assert row.status == AssignmentStatus.RELEASED
    finally:
        await service.stop()


async def test_instance_manager_poll_and_command_loops_continue_after_transient_error(session_factory):
    config = DistributedConfig(
        shard_count=8,
        manager_healthy_ttl=timedelta(seconds=15),
        manager_poll_interval=0.001,
    )
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='im-loop-errors',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={},
    )
    poll_calls = 0
    command_calls = 0

    async def flaky_poll():
        nonlocal poll_calls
        poll_calls += 1
        if poll_calls == 1:
            raise RuntimeError('poll boom')
        service._stopping.set()
        return 0

    async def flaky_command():
        nonlocal command_calls
        command_calls += 1
        if command_calls == 1:
            raise RuntimeError('command boom')
        service._stopping.set()
        return 0

    poll_loop_claimed_commands = False

    async def fail_command_from_poll_loop():
        nonlocal poll_loop_claimed_commands
        poll_loop_claimed_commands = True
        raise AssertionError('poll loop should not claim commands')

    service.poll_assignments_once = flaky_poll
    service.claim_commands_once = fail_command_from_poll_loop
    await service._poll_loop()
    assert poll_calls >= 2
    assert poll_loop_claimed_commands is False

    service._stopping.clear()
    service.claim_commands_once = flaky_command
    await service._command_loop()
    assert command_calls >= 2


@pytest.mark.parametrize('errno', [1053, 1158, 1159, 1160, 1161, 1205, 1213, 2003])
async def test_manager_transaction_retries_transient_mysql_error(errno):
    class TransientDatabaseError(Exception):
        pass

    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OperationalError(
                'UPDATE executions',
                {},
                TransientDatabaseError(errno, 'temporary database failure'),
            )
        return 'ok'

    assert await retry_transient_db_transaction(operation) == 'ok'
    assert attempts == 2


@pytest.mark.parametrize('errno', [1053, 1158, 1159, 1160, 1161, 1205, 1213, 2003, 2013])
async def test_manager_batch_control_retries_retryable_storage_failure(monkeypatch, errno):
    from nexus.distributed.instance_manager import service as service_module

    class DatabaseError(Exception):
        pass

    class Control:
        def __init__(self) -> None:
            self.calls = 0

        async def ensure_prepared(self, parent_execution_id, *, parent_attempt_id=None):
            del parent_execution_id, parent_attempt_id
            self.calls += 1
            if self.calls == 1:
                raise OperationalError('batch control', {}, DatabaseError(errno, 'transient'))
            return 'prepared'

    class Service:
        def __init__(self) -> None:
            self.batch_control = Control()

        async def call_manager_loop(self, operation, *, timeout=30.0):
            del timeout
            return await operation

    monkeypatch.setattr(service_module.asyncio, 'sleep', AsyncMock())
    service = Service()

    result = await ManagerBatchControl(service).ensure_prepared('batch-1', parent_attempt_id='attempt-1')

    assert result == 'prepared'
    assert service.batch_control.calls == 2


async def test_manager_batch_control_maps_retry_exhaustion_to_domain_error(monkeypatch):
    from nexus.distributed.batch.ports import BatchControlTransientError
    from nexus.distributed.instance_manager import service as service_module

    class DatabaseError(Exception):
        pass

    class Control:
        def __init__(self) -> None:
            self.calls = 0

        async def ensure_prepared(self, parent_execution_id, *, parent_attempt_id=None):
            del parent_execution_id, parent_attempt_id
            self.calls += 1
            raise OperationalError('batch control', {}, DatabaseError(1213, 'deadlock'))

    class Service:
        def __init__(self) -> None:
            self.batch_control = Control()

        async def call_manager_loop(self, operation, *, timeout=30.0):
            del timeout
            return await operation

    monkeypatch.setattr(service_module.asyncio, 'sleep', AsyncMock())
    service = Service()

    with pytest.raises(BatchControlTransientError, match='batch_control_storage_retry_exhausted'):
        await ManagerBatchControl(service).ensure_prepared('batch-1', parent_attempt_id='attempt-1')

    assert service.batch_control.calls == 9


async def test_manager_batch_control_maps_permanent_database_error_without_retry():
    from nexus.distributed.batch.ports import BatchControlStorageError

    class DatabaseError(Exception):
        pass

    class Control:
        def __init__(self) -> None:
            self.calls = 0

        async def ensure_prepared(self, parent_execution_id, *, parent_attempt_id=None):
            del parent_execution_id, parent_attempt_id
            self.calls += 1
            raise OperationalError('batch control', {}, DatabaseError(1049, 'unknown database'))

    class Service:
        def __init__(self) -> None:
            self.batch_control = Control()

        async def call_manager_loop(self, operation, *, timeout=30.0):
            del timeout
            return await operation

    service = Service()

    with pytest.raises(BatchControlStorageError, match='batch_control_storage_failed'):
        await ManagerBatchControl(service).ensure_prepared('batch-1', parent_attempt_id='attempt-1')

    assert service.batch_control.calls == 1


async def test_manager_batch_control_preserves_non_database_domain_error():
    class Control:
        async def ensure_prepared(self, parent_execution_id, *, parent_attempt_id=None):
            del parent_execution_id, parent_attempt_id
            raise ValueError('batch_parent_attempt_fenced')

    class Service:
        batch_control = Control()

        async def call_manager_loop(self, operation, *, timeout=30.0):
            del timeout
            return await operation

    with pytest.raises(ValueError, match='batch_parent_attempt_fenced'):
        await ManagerBatchControl(Service()).ensure_prepared('batch-1', parent_attempt_id='attempt-1')


async def test_manager_batch_control_uses_reporter_deadline_for_reconcile():
    deadlines: list[float] = []

    class Control:
        async def ensure_prepared(self, parent_execution_id, *, parent_attempt_id=None, bridge_timeout=None):
            del parent_execution_id, parent_attempt_id, bridge_timeout
            return 'prepared'

        async def reconcile_once(self, parent_execution_id, **kwargs):
            del parent_execution_id, kwargs
            return True

    class Service:
        batch_control = Control()

        async def call_manager_loop(self, operation, *, timeout):
            deadlines.append(timeout)
            return await operation

    control = ManagerBatchControl(Service())

    assert (
        await control.ensure_prepared(
            'batch-1',
            parent_attempt_id='attempt-1',
            bridge_timeout=32.0,
        )
        == 'prepared'
    )
    assert (
        await control.reconcile_once(
            'batch-1',
            max_inflight=10,
            submit_batch_size=5,
            parent_attempt_id='attempt-1',
        )
        is True
    )
    assert deadlines == [32.0, 30.0]


async def test_instance_manager_wakeup_triggers_command_loop_without_waiting_for_sleep(session_factory):
    config = DistributedConfig(
        shard_count=8,
        manager_healthy_ttl=timedelta(seconds=15),
        manager_poll_interval=60,
    )
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='im-command-wakeup',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={},
    )
    command_calls = 0
    first_claim = asyncio.Event()
    second_claim = asyncio.Event()

    async def fake_command():
        nonlocal command_calls
        command_calls += 1
        if command_calls == 1:
            first_claim.set()
        if command_calls == 2:
            second_claim.set()
            service._stopping.set()
        return 0

    service.claim_commands_once = fake_command
    task = asyncio.create_task(service._command_loop())
    try:
        await asyncio.wait_for(first_claim.wait(), timeout=1)
        service.request_wakeup()
        await asyncio.wait_for(second_claim.wait(), timeout=1)
        await asyncio.wait_for(task, timeout=1)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_instance_manager_set_draining_persists_immediately(session_factory):
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(manager_healthy_ttl=timedelta(seconds=15)),
        manager_id='im-draining',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={},
    )
    assert await service._register_session() is True

    assert await service.set_draining(True) is True

    async with session_factory() as session:
        row = await session.get(InstanceManagerRow, service.manager_id)
    assert row is not None
    assert row.session_id == service.session_id
    assert row.draining is True


async def test_instance_manager_new_session_registers_not_draining(session_factory):
    config = DistributedConfig(manager_healthy_ttl=timedelta(seconds=15))
    previous = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='im-restarted',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={},
    )
    assert await previous._register_session() is True
    assert await previous.set_draining(True) is True

    restarted = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id=previous.manager_id,
        endpoint_url=previous.endpoint_url,
        capacity_total=1,
        driver_factories={},
        force_replace=True,
    )
    assert restarted.session_id != previous.session_id
    assert await restarted._register_session() is True

    async with session_factory() as session:
        row = await session.get(InstanceManagerRow, restarted.manager_id)
    assert row is not None
    assert row.session_id == restarted.session_id
    assert row.draining is False


async def test_instance_manager_polls_pending_assignment_and_completes_execution(session_factory):
    config = DistributedConfig(
        shard_count=8,
        manager_healthy_ttl=timedelta(seconds=15),
        manager_poll_interval=0.01,
        manager_heartbeat_interval=0.01,
    )
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='im-1',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={'nexus_single': lambda **_: CompletingDriver()},
    )
    await service.start()
    try:
        async with session_factory() as session:
            execution_repo = ExecutionRepository(session, shard_count=8)
            execution = await execution_repo.create_execution(
                CreateExecution(
                    execution_id='job-1',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={'task_name': 'identity'},
                ),
                now=utcnow(),
            )
            scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
            assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
            await session.commit()

        assert assignment is not None
        await service.poll_assignments_once()
        await service.wait_for_idle(timeout=2)

        async with session_factory() as session:
            current = await ExecutionRepository(session, shard_count=8).get_execution('job-1')

        assert current.status == ExecutionStatus.SUCCEEDED
        assert current.current_assignment.status == AssignmentStatus.RELEASED
    finally:
        await service.stop()


async def test_instance_manager_converges_unhandled_driver_failure_without_execution_ttl(session_factory):
    config = DistributedConfig(
        shard_count=8,
        manager_healthy_ttl=timedelta(seconds=15),
        manager_poll_interval=0.01,
        manager_heartbeat_interval=0.01,
    )
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='im-unhandled-driver',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={'nexus_single': lambda **_: UnhandledFailureDriver()},
    )
    await service.start()
    try:
        async with session_factory() as session:
            execution = await ExecutionRepository(session, shard_count=8).create_execution(
                CreateExecution(
                    execution_id='job-unhandled-driver',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={'task_name': 'identity'},
                    execution_ttl_seconds=None,
                    max_attempts=1,
                ),
                now=utcnow(),
            )
            assignment = await SchedulerRepository(
                session,
                shard_count=8,
                manager_healthy_ttl=timedelta(seconds=15),
            ).claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
            await session.commit()

        assert assignment is not None
        assert await service.poll_assignments_once() == 1
        await service.wait_for_idle(timeout=2)

        async with session_factory() as session:
            current = await ExecutionRepository(session, shard_count=8).get_execution('job-unhandled-driver')
            current_row = await session.get(ExecutionRow, 'job-unhandled-driver')

        assert current_row is not None
        assert current_row.execution_ttl_seconds is None
        assert current.status == ExecutionStatus.FAILED
        assert current.current_assignment.status == AssignmentStatus.RELEASED
        assert current_row.error_json['type'] == 'UnhandledDriverError'
        assert current_row.error_json['cause_type'] == 'RuntimeError'
    finally:
        await service.stop()


async def test_instance_manager_persists_failed_trace_result_pointer(session_factory):
    config = DistributedConfig(
        shard_count=8,
        manager_healthy_ttl=timedelta(seconds=15),
        manager_poll_interval=0.01,
        manager_heartbeat_interval=0.01,
    )
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='im-failed-result',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={'nexus_single': lambda **_: FailingWithResultDriver()},
    )
    await service.start()
    try:
        async with session_factory() as session:
            execution_repo = ExecutionRepository(session, shard_count=8)
            execution = await execution_repo.create_execution(
                CreateExecution(
                    execution_id='job-failed-result',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={'task_name': 'identity'},
                ),
                now=utcnow(),
            )
            scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
            assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
            await session.commit()

        assert assignment is not None
        await service.poll_assignments_once()
        await service.wait_for_idle(timeout=2)

        async with session_factory() as session:
            row = await session.get(ExecutionRow, 'job-failed-result')

        assert row is not None
        assert row.status == ExecutionStatus.FAILED
        assert row.error_json == {'type': 'TaskFailed', 'message': 'nope'}
        assert row.result_json == {
            '_external': True,
            'kind': 'nexus_run_trace_result',
            'metadata': {'version': 4},
        }
    finally:
        await service.stop()


async def test_instance_manager_requeues_assignment_when_executor_start_fails(session_factory):
    config = DistributedConfig(
        shard_count=8,
        manager_healthy_ttl=timedelta(seconds=15),
        manager_poll_interval=0.01,
        manager_heartbeat_interval=0.01,
    )
    executor = FailingStartExecutor()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='im-start-fails',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        executor=executor,
        driver_factories={'nexus_single': lambda **_: CompletingDriver()},
    )
    await service.start()
    try:
        async with session_factory() as session:
            execution_repo = ExecutionRepository(session, shard_count=8)
            execution = await execution_repo.create_execution(
                CreateExecution(
                    execution_id='job-start-fails',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={'task_name': 'identity'},
                    max_attempts=2,
                ),
                now=utcnow(),
            )
            scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
            assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
            await session.commit()

        assert assignment is not None
        await service.poll_assignments_once()
        await asyncio.wait_for(executor.started.wait(), timeout=2)

        async with session_factory() as session:
            current = await ExecutionRepository(session, shard_count=8).get_execution('job-start-fails')

        assert current.status == ExecutionStatus.QUEUED
        assert current.attempt_count == 1
        assert current.current_assignment.status == AssignmentStatus.LOST
        assert not service._running
    finally:
        await service.stop()


async def test_instance_manager_keeps_assignment_when_timed_out_start_is_present_in_executor(session_factory) -> None:
    class TimedOutStartExecutor(FailingStartExecutor):
        async def has_assignment(self, assignment_id):  # noqa: ARG002
            return True

        async def start_assignment(self, **kwargs):  # noqa: ARG002
            self.started.set()
            raise TimeoutError('executor bridge timed out')

    executor = TimedOutStartExecutor()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(aiomonitor_enabled=False, shard_count=8),
        manager_id='im-start-timeout-present',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        executor=executor,
        driver_factories={'nexus_single': lambda **_: CompletingDriver()},
    )
    await service.start()
    try:
        async with session_factory() as session:
            execution_repo = ExecutionRepository(session, shard_count=8)
            execution = await execution_repo.create_execution(
                CreateExecution(
                    execution_id='job-start-timeout-present',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={'task_name': 'identity'},
                    max_attempts=2,
                ),
                now=utcnow(),
            )
            assignment = await SchedulerRepository(
                session,
                shard_count=8,
                manager_healthy_ttl=timedelta(seconds=15),
            ).claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
            await session.commit()

        assert assignment is not None
        assert await service.poll_assignments_once() == 1
        assert assignment.assignment_id in service._running
        async with session_factory() as session:
            current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)
            assert current.status == ExecutionStatus.STARTING
            assert current.current_assignment is not None
            assert current.current_assignment.status == AssignmentStatus.ACTIVE
    finally:
        await service.stop()


async def test_instance_manager_registers_thread_executor_instance_mode_and_capacity(session_factory):
    config = DistributedConfig(
        shard_count=8,
        manager_healthy_ttl=timedelta(seconds=15),
        manager_poll_interval=0.01,
        manager_heartbeat_interval=0.01,
    )
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='im-thread',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=2,
        executor=ThreadExecutor(strategy='per_execution', max_threads=2),
        driver_factories={'nexus_single': lambda **_: CompletingDriver()},
    )
    await service.start()
    try:
        async with session_factory() as session:
            manager = await session.get(InstanceManagerRow, 'im-thread')

        assert manager is not None
        assert manager.supported_instance_modes_json == ['thread']
        assert manager.capacity_total == 2
    finally:
        await service.stop()


async def test_instance_manager_control_loop_never_reads_executor_capacity(session_factory) -> None:
    class CapacityTrapExecutor(LoggingExecutor):
        async def capacity_snapshot(self):
            raise AssertionError('Manager control loop must use its local capacity ledger')

    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(aiomonitor_enabled=False),
        manager_id='im-local-capacity-ledger',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=3,
        executor=CapacityTrapExecutor(capacity_total=3),
        driver_factories={},
    )
    await service.start()
    try:
        assert await service.heartbeat_once() is True
        assert await service.poll_assignments_once() == 0

        async with session_factory() as session:
            manager = await session.get(InstanceManagerRow, service.manager_id)
            assert manager is not None
            assert manager.capacity_total == 3
    finally:
        await service.stop()


async def test_instance_manager_heartbeat_survives_a_blocked_default_executor_loop(
    session_factory,
    monkeypatch,
) -> None:
    release = threading.Event()
    entered = threading.Event()

    class SynchronouslyBlockingDriver(BlockingDriver):
        async def start(self, handle, spec, config, callbacks):  # noqa: ARG002
            entered.set()
            release.wait()

    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(aiomonitor_enabled=False, shard_count=8),
        manager_id='im-blocked-default-executor',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={'nexus_single': lambda **_: SynchronouslyBlockingDriver()},
    )
    monkeypatch.setattr(
        'nexus.distributed.instance_manager.service._EXECUTOR_PROBE_TIMEOUT_SECONDS',
        0.02,
    )
    await service.start()
    try:
        async with session_factory() as session:
            execution_repo = ExecutionRepository(session, shard_count=8)
            execution = await execution_repo.create_execution(
                CreateExecution(
                    execution_id='job-blocked-default-executor',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={'task_name': 'identity'},
                ),
                now=utcnow(),
            )
            assignment = await SchedulerRepository(
                session,
                shard_count=8,
                manager_healthy_ttl=timedelta(seconds=15),
            ).claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
            await session.commit()

        assert assignment is not None
        assert await service.poll_assignments_once() == 1
        assert await asyncio.to_thread(entered.wait, 1)

        assert await asyncio.wait_for(service.heartbeat_once(), timeout=0.5) is True
        health = await service._probe_executor_once(now=10.0)

        assert health.responsive is False
        assert service._local_capacity_snapshot().capacity_total == 0
        async with session_factory() as session:
            liveness = await session.get(InstanceManagerLivenessRow, service.manager_id)
        assert liveness is not None
        assert liveness.last_heartbeat_at is not None
    finally:
        release.set()
        await service.stop()


async def test_instance_manager_allows_ten_seconds_for_executor_health_probe(session_factory) -> None:
    probe_timeouts: list[float] = []

    class RecordingExecutor(LoggingExecutor):
        async def probe_health(self, *, timeout, hang_timeout, now):  # noqa: ARG002
            probe_timeouts.append(timeout)
            return ExecutorHealthSnapshot(responsive=True, effective_capacity=1)

    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(aiomonitor_enabled=False),
        manager_id='im-executor-probe-timeout',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        executor=RecordingExecutor(),
        driver_factories={},
    )
    try:
        await service._probe_executor_once(now=10.0)
    finally:
        await service.stop()

    assert probe_timeouts == [10.0]


async def test_instance_manager_executor_probe_suppresses_capacity_until_recovery(session_factory) -> None:
    class RecoveringExecutor(LoggingExecutor):
        def __init__(self) -> None:
            super().__init__(capacity_total=3)
            self.responses = [False, True]

        async def probe_health(self, *, timeout, hang_timeout, now):  # noqa: ARG002
            responsive = self.responses.pop(0)
            return ExecutorHealthSnapshot(
                responsive=responsive,
                effective_capacity=3 if responsive else 0,
                unresponsive_workers=0 if responsive else 1,
            )

    executor = RecoveringExecutor()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(aiomonitor_enabled=False),
        manager_id='im-executor-recovers',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=3,
        executor=executor,
        driver_factories={},
    )
    await service.start()
    try:
        await service._probe_executor_once(now=10.0)
        assert service._local_capacity_snapshot().capacity_total == 0
        assert await service.heartbeat_once() is True

        async with session_factory() as session:
            liveness = await session.get(InstanceManagerLivenessRow, service.manager_id)
            assert liveness is not None
            assert liveness.capacity_total == 0

        await service._probe_executor_once(now=11.0)
        assert service._local_capacity_snapshot().capacity_total == 3
    finally:
        await service.stop()


async def test_instance_manager_reconciles_before_reenabling_an_executor_with_unknown_start_state(
    session_factory,
) -> None:
    class HealthyExecutor(LoggingExecutor):
        async def probe_health(self, *, timeout, hang_timeout, now):  # noqa: ARG002
            return ExecutorHealthSnapshot(responsive=True, effective_capacity=1)

    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(aiomonitor_enabled=False),
        manager_id='im-executor-unknown-start-state',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        executor=HealthyExecutor(),
        driver_factories={},
    )
    service._mark_executor_unavailable()
    service._reconcile_executor_assignments_after_recovery = AsyncMock()

    await service._probe_executor_once(now=10.0)

    service._reconcile_executor_assignments_after_recovery.assert_awaited_once_with()
    assert service._executor_health.responsive is True


async def test_instance_manager_keeps_capacity_closed_when_recovery_reconciliation_fails(session_factory) -> None:
    class HealthyExecutor(LoggingExecutor):
        async def probe_health(self, *, timeout, hang_timeout, now):  # noqa: ARG002
            return ExecutorHealthSnapshot(responsive=True, effective_capacity=2)

    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(aiomonitor_enabled=False),
        manager_id='im-executor-reconcile-fails',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=2,
        executor=HealthyExecutor(capacity_total=2),
        driver_factories={},
    )
    service._mark_executor_unavailable()
    service._reconcile_executor_assignments_after_recovery = AsyncMock(side_effect=RuntimeError('reconcile failed'))

    health = await service._probe_executor_once(now=10.0)

    assert health.responsive is False
    assert service._local_capacity_snapshot().capacity_total == 0


async def test_instance_manager_reconciles_missing_assignment_before_executor_recovery(
    session_factory,
    tmp_path,
) -> None:
    class RecoveringStartExecutor(LoggingExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.assignment_state_unknown = True
            self.responses = [False, True]

        async def start_assignment(self, **kwargs):  # noqa: ARG002
            self.started.set()
            raise TimeoutError('executor bridge timed out')

        async def has_assignment(self, assignment_id):  # noqa: ARG002
            if self.assignment_state_unknown:
                raise TimeoutError('executor assignment state unavailable')
            return False

        async def probe_health(self, *, timeout, hang_timeout, now):  # noqa: ARG002
            responsive = self.responses.pop(0)
            if responsive:
                self.assignment_state_unknown = False
            return ExecutorHealthSnapshot(
                responsive=responsive,
                effective_capacity=1 if responsive else 0,
                unresponsive_workers=0 if responsive else 1,
            )

    executor = RecoveringStartExecutor()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(aiomonitor_enabled=False, shard_count=8),
        manager_id='im-executor-recovery-reconcile',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        executor=executor,
        driver_factories={'nexus_single': lambda **_: CompletingDriver()},
    )
    await service.start()
    try:
        async with session_factory() as session:
            execution_repo = ExecutionRepository(session, shard_count=8)
            execution = await execution_repo.create_execution(
                CreateExecution(
                    execution_id='job-executor-recovery-reconcile',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={'task_name': 'identity'},
                    max_attempts=2,
                ),
                now=utcnow(),
            )
            assignment = await SchedulerRepository(
                session,
                shard_count=8,
                manager_healthy_ttl=timedelta(seconds=15),
            ).claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
            await session.commit()

        assert assignment is not None
        assert await service.poll_assignments_once() == 1
        assert assignment.assignment_id in service._running
        temporary = _temporary_task_files(tmp_path, assignment.assignment_id)
        service._running[assignment.assignment_id].temporary_task_files = temporary

        await service._probe_executor_once(now=10.0)
        assert service._local_capacity_snapshot().capacity_total == 0

        await service._probe_executor_once(now=11.0)

        assert assignment.assignment_id not in service._running
        assert not temporary.directory.exists()
        assert service._local_capacity_snapshot().capacity_total == 1
        async with session_factory() as session:
            current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)
        assert current.status == ExecutionStatus.QUEUED
        assert current.current_assignment is not None
        assert current.current_assignment.status == AssignmentStatus.LOST
    finally:
        await service.stop()


async def test_instance_manager_self_fences_when_batch_blocking_helper_times_out(session_factory) -> None:
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(
            aiomonitor_enabled=False,
            manager_executor_hang_timeout=timedelta(milliseconds=20),
        ),
        manager_id='im-batch-blocking-timeout',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        workload='orchestration',
    )
    release = threading.Event()
    reasons: list[str] = []

    async def capture_self_fence(reason: str) -> None:
        reasons.append(reason)

    service._self_fence = capture_self_fence
    try:
        with pytest.raises(TimeoutError, match='batch_blocking_operation_timeout:test-helper'):
            await service.batch_control._run_blocking('test-helper', release.wait)
        assert reasons == ['batch_blocking_timeout:test-helper']
    finally:
        release.set()
        await asyncio.sleep(0.05)
        await service.batch_control.close()


async def test_instance_manager_self_fences_after_continuous_executor_hang(session_factory) -> None:
    class HungExecutor(LoggingExecutor):
        async def probe_health(self, *, timeout, hang_timeout, now):  # noqa: ARG002
            return ExecutorHealthSnapshot(responsive=False, effective_capacity=0, unresponsive_workers=1)

    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(
            aiomonitor_enabled=False,
            manager_executor_hang_timeout=timedelta(seconds=3),
        ),
        manager_id='im-executor-hung',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        executor=HungExecutor(),
        driver_factories={},
    )
    reasons: list[str] = []

    async def capture_self_fence(reason: str) -> None:
        reasons.append(reason)

    service._self_fence = capture_self_fence

    await service._probe_executor_once(now=20.0)
    await service._probe_executor_once(now=22.9)
    assert reasons == []

    await service._probe_executor_once(now=23.0)
    assert reasons == ['executor_loop_hang']


async def test_instance_manager_starts_executor_watchdog_with_other_control_loops(session_factory) -> None:
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(aiomonitor_enabled=False),
        manager_id='im-executor-watchdog',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        executor=LoggingExecutor(),
        driver_factories={},
    )
    probed = asyncio.Event()

    async def probe_once(*, now=None):  # noqa: ARG001
        probed.set()
        service._stopping.set()
        return ExecutorHealthSnapshot(responsive=True, effective_capacity=1)

    async def heartbeat_once() -> bool:
        return True

    async def no_work() -> int:
        return 0

    async def update_snapshot() -> bool:
        return True

    service._probe_executor_once = probe_once
    service.heartbeat_once = heartbeat_once
    service.poll_assignments_once = no_work
    service.claim_commands_once = no_work
    service.reconcile_local_assignments_once = no_work
    service.update_local_snapshot_once = update_snapshot
    await service.start()
    service.start_background_tasks()
    try:
        await asyncio.wait_for(probed.wait(), timeout=1)
        assert service._executor_watchdog_task is not None
    finally:
        await asyncio.wait_for(service.stop(), timeout=1)


async def test_instance_manager_quarantines_one_thread_assignment_without_restarting_at_half_capacity(
    session_factory,
    tmp_path,
) -> None:
    class QuarantiningExecutor(LoggingExecutor):
        instance_mode = 'thread'

        def __init__(self) -> None:
            super().__init__(capacity_total=2)
            self.quarantined: QuarantinedAssignment | None = None

        async def probe_health(self, *, timeout, hang_timeout, now):  # noqa: ARG002
            events = (self.quarantined,) if self.quarantined is not None else ()
            self.quarantined = None
            return ExecutorHealthSnapshot(
                responsive=True,
                effective_capacity=2 - len(events),
                unresponsive_workers=len(events),
                quarantined_assignments=events,
                recycle_requested=bool(events),
            )

    executor = QuarantiningExecutor()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(aiomonitor_enabled=False, shard_count=8),
        manager_id='im-thread-quarantine',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=2,
        executor=executor,
        driver_factories={'nexus_single': lambda **_: BlockingDriver()},
    )
    reasons: list[str] = []

    async def capture_self_fence(reason: str) -> None:
        reasons.append(reason)

    service._self_fence = capture_self_fence
    await service.start()
    try:
        async with session_factory() as session:
            execution_repo = ExecutionRepository(session, shard_count=8)
            executions = [
                await execution_repo.create_execution(
                    CreateExecution(
                        execution_id=f'thread-quarantine-{index}',
                        mode=ExecutionMode.JOB,
                        spec_type='nexus_single',
                        instance_mode='thread',
                        spec={'task_name': 'identity'},
                        max_attempts=2,
                    ),
                    now=utcnow(),
                )
                for index in range(2)
            ]
            scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
            assignments = [
                await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
                for execution in executions
            ]
            await session.commit()

        assert all(assignment is not None for assignment in assignments)
        assert await service.poll_assignments_once() == 2
        first = assignments[0]
        assert first is not None
        temporary = _temporary_task_files(tmp_path, first.assignment_id)
        service._running[first.assignment_id].temporary_task_files = temporary
        executor.quarantined = QuarantinedAssignment(
            assignment_id=first.assignment_id,
            execution_id=first.execution_id,
            worker_id='1',
            slot_cost=1,
        )

        await service._probe_executor_once(now=10.0)

        assert reasons == []
        assert first.assignment_id not in service._running
        assert not temporary.directory.exists()
        assert len(service._running) == 1
        capacity = service._local_capacity_snapshot()
        assert capacity.capacity_total == 1
        assert capacity.running_slot_cost == 1
        assert capacity.available_slots == 0
        async with session_factory() as session:
            lost = await ExecutionRepository(session, shard_count=8).get_execution(first.execution_id)
            assert lost.status == ExecutionStatus.QUEUED
            assert lost.current_assignment is not None
            assert lost.current_assignment.status == AssignmentStatus.LOST

        with pytest.raises(ValueError, match='assignment_not_active'):
            await ManagerExecutionReporter(service, first).mark_running()

        executor.quarantined = QuarantinedAssignment(
            assignment_id=first.assignment_id,
            execution_id=first.execution_id,
            worker_id='1',
            slot_cost=1,
        )
        await service._probe_executor_once(now=11.0)
        assert service._local_capacity_snapshot().capacity_total == 1
    finally:
        await service.stop()


async def test_instance_manager_restarts_only_below_thread_capacity_threshold(session_factory) -> None:
    class DegradedExecutor(LoggingExecutor):
        instance_mode = 'thread'

        def __init__(self) -> None:
            super().__init__(capacity_total=4)
            self.effective_capacity = 4

        async def probe_health(self, *, timeout, hang_timeout, now):  # noqa: ARG002
            return ExecutorHealthSnapshot(
                responsive=True,
                effective_capacity=self.effective_capacity,
                recycle_requested=self.effective_capacity < self.capacity_total,
            )

    executor = DegradedExecutor()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(aiomonitor_enabled=False),
        manager_id='im-thread-capacity-threshold',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=4,
        executor=executor,
        driver_factories={},
    )
    service._running['healthy-assignment'] = RunningAssignment(
        execution_id='healthy-execution',
        assignment_id='healthy-assignment',
        slot_cost=1,
        stop_timeout_seconds=1,
        stop_disposition=StopDisposition.IMMEDIATE,
    )
    reasons: list[str] = []

    async def capture_self_fence(reason: str) -> None:
        reasons.append(reason)

    service._self_fence = capture_self_fence
    executor.effective_capacity = 2

    await service._probe_executor_once(now=10.0)
    assert service._local_capacity_snapshot().capacity_total == 2
    assert reasons == []

    executor.effective_capacity = 1
    await service._probe_executor_once(now=11.0)
    assert service._local_capacity_snapshot().capacity_total == 1
    assert reasons == ['thread_capacity_below_threshold']


async def test_instance_manager_fences_below_thread_capacity_threshold_before_repository_replay(
    session_factory,
    monkeypatch,
) -> None:
    class DegradedExecutor(LoggingExecutor):
        instance_mode = 'thread'

        async def probe_health(self, *, timeout, hang_timeout, now):  # noqa: ARG002
            return ExecutorHealthSnapshot(
                responsive=True,
                effective_capacity=0,
                unresponsive_workers=1,
                quarantined_assignments=(
                    QuarantinedAssignment(
                        assignment_id='assignment-replay-blocked',
                        execution_id='execution-replay-blocked',
                        worker_id='1',
                        slot_cost=1,
                    ),
                ),
                recycle_requested=True,
            )

    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(aiomonitor_enabled=False),
        manager_id='im-thread-capacity-replay-blocked',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        executor=DegradedExecutor(capacity_total=1),
        driver_factories={},
    )
    service._running['assignment-replay-blocked'] = RunningAssignment(
        execution_id='execution-replay-blocked',
        assignment_id='assignment-replay-blocked',
        slot_cost=1,
        stop_timeout_seconds=1,
        stop_disposition=StopDisposition.IMMEDIATE,
    )
    replay_release = asyncio.Event()
    replay_entered = asyncio.Event()
    fenced = asyncio.Event()

    async def blocked_replay(self, *args, **kwargs):  # noqa: ARG001
        replay_entered.set()
        await replay_release.wait()
        return False

    async def capture_self_fence(reason: str) -> None:
        assert reason == 'thread_capacity_below_threshold'
        fenced.set()

    monkeypatch.setattr(ManagerRepository, 'mark_assignment_lost', blocked_replay)
    service._self_fence = capture_self_fence
    probe = asyncio.create_task(service._probe_executor_once(now=10.0))
    try:
        await asyncio.wait_for(replay_entered.wait(), timeout=1)
        await asyncio.sleep(0.05)
        assert fenced.is_set()
    finally:
        replay_release.set()
        await probe


async def test_instance_manager_recycles_quarantined_threads_after_healthy_work_drains(session_factory) -> None:
    class DegradedExecutor(LoggingExecutor):
        instance_mode = 'thread'

        async def probe_health(self, *, timeout, hang_timeout, now):  # noqa: ARG002
            return ExecutorHealthSnapshot(
                responsive=True,
                effective_capacity=1,
                unresponsive_workers=1,
                quarantined_assignments=(
                    QuarantinedAssignment(
                        assignment_id=None,
                        execution_id=None,
                        worker_id='1',
                        slot_cost=1,
                    ),
                ),
                recycle_requested=True,
            )

    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(aiomonitor_enabled=False),
        manager_id='im-thread-drained',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=2,
        executor=DegradedExecutor(capacity_total=2),
        driver_factories={},
    )
    service._running['healthy-assignment'] = RunningAssignment(
        execution_id='healthy-execution',
        assignment_id='healthy-assignment',
        slot_cost=1,
        stop_timeout_seconds=1,
        stop_disposition=StopDisposition.IMMEDIATE,
    )
    reasons: list[str] = []

    async def capture_self_fence(reason: str) -> None:
        reasons.append(reason)

    service._self_fence = capture_self_fence

    await service._probe_executor_once(now=10.0)
    assert reasons == []

    service._running.clear()
    await service._probe_executor_once(now=11.0)
    assert reasons == ['thread_quarantine_drained']


async def test_instance_manager_fencing_stops_local_execution_and_clears_assignment(session_factory):
    config = DistributedConfig(
        shard_count=8,
        manager_healthy_ttl=timedelta(seconds=15),
        manager_poll_interval=0.01,
        manager_heartbeat_interval=0.01,
    )
    driver = BlockingDriver()
    # Exercise fencing without allowing slow CI cleanup to terminate pytest.
    hard_exit_guard = Mock(spec=HardExitGuard)
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='im-1',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={'nexus_single': lambda **_: driver},
        hard_exit_guard=hard_exit_guard,
    )
    await service.start()
    try:
        async with session_factory() as session:
            execution_repo = ExecutionRepository(session, shard_count=8)
            execution = await execution_repo.create_execution(
                CreateExecution(
                    execution_id='job-fenced',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={'task_name': 'identity'},
                ),
                now=utcnow(),
            )
            scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
            assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
            await session.commit()

        assert assignment is not None
        await service.poll_assignments_once()
        await asyncio.wait_for(driver.started.wait(), timeout=2)

        async with session_factory() as session:
            from nexus.distributed.repositories.manager import ManagerRepository

            manager = ManagerRepository(session, healthy_ttl=timedelta(seconds=15))
            replaced = await manager.register_session(
                manager_id='im-1',
                session_id='replacement-session',
                endpoint_url='http://127.0.0.1:9002',
                supported_modes=[ExecutionMode.JOB],
                supported_spec_types=['nexus_single'],
                supported_instance_modes=['asyncio'],
                capacity_total=1,
                force_replace=True,
                now=utcnow(),
            )
            await session.commit()

        assert replaced is True
        assert await service.heartbeat_once() is False
        hard_exit_guard.arm.assert_called_once_with('heartbeat_fenced')
        await asyncio.wait_for(driver.stopped.wait(), timeout=2)
        await asyncio.wait_for(driver.cancelled.wait(), timeout=2)
        await service.wait_for_idle(timeout=0.1)
    finally:
        await service.stop()

    hard_exit_guard.disarm.assert_called_once_with()


async def test_instance_manager_self_fence_arms_hard_exit_before_executor_cleanup(session_factory, tmp_path) -> None:
    actions: list[tuple[str, str | None]] = []

    class Guard:
        def arm(self, reason: str) -> None:
            actions.append(('arm', reason))

        def disarm(self) -> None:
            actions.append(('disarm', None))

    class GuardAwareExecutor(LoggingExecutor):
        async def stop_assignment(self, assignment_id, timeout=10.0):  # noqa: ARG002
            assert actions == [('arm', 'executor_loop_hang')]
            self._assignments.pop(assignment_id, None)

    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(aiomonitor_enabled=False),
        manager_id='im-hard-exit-order',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        executor=GuardAwareExecutor(),
        driver_factories={},
    )
    service._hard_exit_guard = Guard()
    service._hard_exit_enabled = True
    temporary = _temporary_task_files(tmp_path, 'assignment-hard-exit-order')
    service._running['assignment-hard-exit-order'] = RunningAssignment(
        execution_id='execution-hard-exit-order',
        assignment_id='assignment-hard-exit-order',
        slot_cost=1,
        stop_timeout_seconds=1,
        stop_disposition=StopDisposition.IMMEDIATE,
        temporary_task_files=temporary,
    )

    await service._self_fence('executor_loop_hang')

    assert actions == [('arm', 'executor_loop_hang')]
    assert not service._running
    assert not temporary.directory.exists()


async def test_instance_manager_reconciles_local_orphan_assignment(session_factory, tmp_path):
    config = DistributedConfig(
        shard_count=8,
        manager_healthy_ttl=timedelta(seconds=15),
        manager_poll_interval=0.01,
        manager_heartbeat_interval=0.01,
    )
    driver = BlockingDriver()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='im-orphan',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={'nexus_single': lambda **_: driver},
    )
    await service.start()
    try:
        async with session_factory() as session:
            execution_repo = ExecutionRepository(session, shard_count=8)
            execution = await execution_repo.create_execution(
                CreateExecution(
                    execution_id='job-local-orphan',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={'task_name': 'identity'},
                ),
                now=utcnow(),
            )
            scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
            assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
            await session.commit()

        assert assignment is not None
        await service.poll_assignments_once()
        await asyncio.wait_for(driver.started.wait(), timeout=2)
        temporary = _temporary_task_files(tmp_path, assignment.assignment_id)
        service._running[assignment.assignment_id].temporary_task_files = temporary

        async with session_factory() as session:
            assignment_row = await session.get(AssignmentRow, assignment.assignment_id)
            execution_row = await session.get(ExecutionRow, execution.execution_id)
            assert assignment_row is not None
            assert execution_row is not None
            assignment_row.status = AssignmentStatus.LOST
            assignment_row.finished_at = utcnow()
            assignment_row.updated_at = utcnow()
            execution_row.status = ExecutionStatus.QUEUED
            execution_row.current_assignment_id = None
            execution_row.current_attempt_id = None
            await session.commit()

        assert await service.reconcile_local_assignments_once() == 1
        await asyncio.wait_for(driver.stopped.wait(), timeout=2)
        await asyncio.wait_for(driver.cancelled.wait(), timeout=2)
        await service.wait_for_idle(timeout=0.1)
        assert not temporary.directory.exists()
    finally:
        await service.stop()


async def test_instance_manager_retains_stale_reservation_when_executor_stop_state_is_unknown(
    session_factory,
    tmp_path,
    monkeypatch,
):
    driver = BlockingDriver()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15)),
        manager_id='im-orphan-stop-timeout',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={'nexus_single': lambda **_: driver},
    )
    await service.start()
    original_stop = service.executor.stop_assignment
    try:
        async with session_factory() as session:
            execution = await ExecutionRepository(session, shard_count=8).create_execution(
                CreateExecution(
                    execution_id='job-local-orphan-stop-timeout',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={'task_name': 'identity'},
                ),
                now=utcnow(),
            )
            assignment = await SchedulerRepository(
                session,
                shard_count=8,
                manager_healthy_ttl=timedelta(seconds=15),
            ).claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
            await session.commit()

        assert assignment is not None
        assert await service.poll_assignments_once() == 1
        await asyncio.wait_for(driver.started.wait(), timeout=2)
        temporary = _temporary_task_files(tmp_path, assignment.assignment_id)
        service._running[assignment.assignment_id].temporary_task_files = temporary

        async with session_factory() as session:
            assignment_row = await session.get(AssignmentRow, assignment.assignment_id)
            execution_row = await session.get(ExecutionRow, execution.execution_id)
            assert assignment_row is not None
            assert execution_row is not None
            assignment_row.status = AssignmentStatus.LOST
            execution_row.status = ExecutionStatus.QUEUED
            execution_row.current_assignment_id = None
            execution_row.current_attempt_id = None
            await session.commit()

        async def stop_times_out(_assignment_id: str, timeout: float = 10.0) -> None:  # noqa: ARG001
            raise TimeoutError

        monkeypatch.setattr(service.executor, 'stop_assignment', stop_times_out)

        assert await service.reconcile_local_assignments_once() == 0
        assert assignment.assignment_id in service._running
        assert await service.executor.has_assignment(assignment.assignment_id)
        assert service._local_capacity_snapshot().available_slots == 0
        assert temporary.directory.exists()
    finally:
        monkeypatch.setattr(service.executor, 'stop_assignment', original_stop)
        await service.stop()


async def test_instance_manager_heartbeat_does_not_run_local_reconcile(session_factory):
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='im-heartbeat-only',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={},
    )
    async with session_factory() as session:
        manager = ManagerRepository(session, healthy_ttl=timedelta(seconds=15))
        capacity = await service.executor.capacity_snapshot()
        await manager.register_session(
            manager_id=service.manager_id,
            session_id=service.session_id,
            endpoint_url=service.endpoint_url,
            supported_modes=[ExecutionMode.JOB, ExecutionMode.LIVE],
            supported_spec_types=service.supported_spec_types,
            supported_instance_modes=[service.executor.instance_mode],
            capacity_total=capacity.capacity_total,
            force_replace=False,
            now=utcnow(),
        )
        await session.commit()

    async def fail_reconcile():
        raise AssertionError('heartbeat should not reconcile local assignments')

    service.reconcile_local_assignments_once = fail_reconcile

    assert await service.heartbeat_once() is True


async def test_instance_manager_reconciles_local_assignments_with_batch_queries(session_factory, monkeypatch):
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    driver = BlockingDriver()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='im-batch-reconcile',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={'nexus_single': lambda **_: driver},
    )
    await service.start()
    try:
        async with session_factory() as session:
            execution_repo = ExecutionRepository(session, shard_count=8)
            execution = await execution_repo.create_execution(
                CreateExecution(
                    execution_id='job-local-batch-reconcile',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={'task_name': 'identity'},
                ),
                now=utcnow(),
            )
            scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
            assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
            await session.commit()

        assert assignment is not None
        await service.poll_assignments_once()
        await asyncio.wait_for(driver.started.wait(), timeout=2)

        async with session_factory() as session:
            assignment_row = await session.get(AssignmentRow, assignment.assignment_id)
            execution_row = await session.get(ExecutionRow, execution.execution_id)
            assert assignment_row is not None
            assert execution_row is not None
            assignment_row.status = AssignmentStatus.LOST
            assignment_row.finished_at = utcnow()
            assignment_row.updated_at = utcnow()
            execution_row.status = ExecutionStatus.QUEUED
            execution_row.current_assignment_id = None
            execution_row.current_attempt_id = None
            await session.commit()

        get_calls = 0
        original_get = AsyncSession.get

        async def counting_get(self, *args, **kwargs):
            nonlocal get_calls
            get_calls += 1
            return await original_get(self, *args, **kwargs)

        monkeypatch.setattr(AsyncSession, 'get', counting_get)

        statements: list[str] = []
        engine = session_factory.kw['bind']

        def capture_sql(_conn, _cursor, statement, _parameters, _context, _executemany):
            statements.append(statement)

        event.listen(engine.sync_engine, 'before_cursor_execute', capture_sql)
        try:
            assert await service.reconcile_local_assignments_once() == 1
        finally:
            event.remove(engine.sync_engine, 'before_cursor_execute', capture_sql)

        assert get_calls == 0
        assignment_select = next(
            statement.lower()
            for statement in statements
            if 'from assignments' in statement.lower() and 'assignments.assignment_id in' in statement.lower()
        )
        execution_select = next(
            statement.lower()
            for statement in statements
            if 'from executions' in statement.lower() and 'executions.execution_id in' in statement.lower()
        )
        assert 'assignments.runtime_info_json' not in assignment_select
        assert 'assignments.trace_carrier_json' not in assignment_select
        assert 'executions.spec_json' not in execution_select
        assert 'executions.driver_config_json' not in execution_select
        await asyncio.wait_for(driver.stopped.wait(), timeout=2)
    finally:
        await service.stop()


async def test_instance_manager_acks_pending_start_batch_in_one_transaction(session_factory, monkeypatch):
    config = DistributedConfig(
        shard_count=8,
        manager_healthy_ttl=timedelta(seconds=15),
        manager_start_batch_size=2,
    )
    executor = LoggingExecutor(capacity_total=5)
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='im-start-batch',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=5,
        executor=executor,
        driver_factories={'nexus_single': lambda **_: CompletingDriver()},
    )
    await service.start()
    try:
        async with session_factory() as session:
            execution_repo = ExecutionRepository(session, shard_count=8)
            scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
            assignments = []
            for idx in range(5):
                execution = await execution_repo.create_execution(
                    CreateExecution(
                        execution_id=f'job-start-batch-{idx}',
                        mode=ExecutionMode.JOB,
                        spec_type='nexus_single',
                        instance_mode='asyncio',
                        spec={'task_name': 'identity'},
                    ),
                    now=utcnow(),
                )
                assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
                assert assignment is not None
                assignments.append(assignment)
            await session.commit()

        commit_count = 0
        original_commit = AsyncSession.commit

        async def counting_commit(self):
            nonlocal commit_count
            commit_count += 1
            return await original_commit(self)

        monkeypatch.setattr(AsyncSession, 'commit', counting_commit)
        started = await service.poll_assignments_once()

        assert started == 2
        assert commit_count == 1
        async with session_factory() as session:
            statuses = (
                await session.scalars(
                    select(AssignmentRow.status).where(
                        AssignmentRow.assignment_id.in_([assignment.assignment_id for assignment in assignments])
                    )
                )
            ).all()
        assert statuses.count(AssignmentStatus.PENDING) == 3
    finally:
        await service.stop()


async def test_instance_manager_stop_drains_pending_manager_loop_callbacks(session_factory):
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    driver = MarkRunningDriver()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='im-drain-callbacks',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={'nexus_live': lambda **_: driver},
    )
    await service.start()
    mark_running_started = asyncio.Event()
    release_mark_running = asyncio.Event()

    async def slow_mark_running(*args, **kwargs):
        mark_running_started.set()
        await release_mark_running.wait()

    service._mark_running = slow_mark_running
    async with session_factory() as session:
        execution_repo = ExecutionRepository(session, shard_count=8)
        execution = await execution_repo.create_execution(
            CreateExecution(
                execution_id='live-drain-callbacks',
                mode=ExecutionMode.LIVE,
                spec_type='nexus_live',
                instance_mode='asyncio',
                spec={'task_name': 'identity'},
                route_status=RouteStatus.ACTIVE,
            ),
            now=utcnow(),
        )
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        await session.commit()

    assert assignment is not None
    await service.poll_assignments_once()
    await asyncio.wait_for(driver.started.wait(), timeout=2)
    await asyncio.wait_for(mark_running_started.wait(), timeout=2)

    stop_task = asyncio.create_task(service.stop())
    await asyncio.sleep(0)
    assert not stop_task.done()

    release_mark_running.set()
    await asyncio.wait_for(stop_task, timeout=2)
    await asyncio.wait_for(driver.cancelled.wait(), timeout=2)


async def test_instance_manager_marks_command_failed_when_driver_command_fails(session_factory):
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    driver = FailingCommandDriver()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='im-1',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={'nexus_live': lambda **_: driver},
    )
    await service.start()
    try:
        async with session_factory() as session:
            execution_repo = ExecutionRepository(session, shard_count=8)
            execution = await execution_repo.create_execution(
                CreateExecution(
                    execution_id='live-command-fails',
                    mode=ExecutionMode.LIVE,
                    spec_type='nexus_live',
                    instance_mode='asyncio',
                    spec={'task_name': 'identity'},
                    route_status=RouteStatus.ACTIVE,
                ),
                now=utcnow(),
            )
            scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
            assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
            await session.commit()

        assert assignment is not None
        await service.poll_assignments_once()
        await asyncio.wait_for(driver.started.wait(), timeout=2)

        async with session_factory() as session:
            manager = ManagerRepository(session)
            await manager.mark_running(
                assignment.assignment_id,
                assignment.manager_id,
                assignment.manager_session_id,
                endpoint_url='http://127.0.0.1:9001/v1/internal/executions/live-command-fails',
                endpoint_ready=True,
                runtime_info={},
                now=utcnow(),
            )
            await manager.mark_waiting_input(
                assignment.assignment_id,
                assignment.manager_id,
                assignment.manager_session_id,
                payload={'prompt': 'choose'},
                now=utcnow(),
            )
            gateway = GatewayRepository(session, manager_healthy_ttl=timedelta(seconds=15))
            command = await gateway.submit_command(
                'live-command-fails',
                command_type='input.submit',
                payload={'text': 'Alpha'},
                idempotency_key='failing-command',
                now=utcnow(),
            )
            await session.commit()

        assert await service.claim_commands_once() == 1

        async with session_factory() as session:
            commands = await ExecutionRepository(session, shard_count=8).list_commands('live-command-fails')

        assert commands[0].command_id == command.command_id
        assert commands[0].status == CommandStatus.FAILED
    finally:
        await service.stop()


async def test_instance_manager_command_logs_include_distributed_execution_context(session_factory, tmp_path):
    distributed_logs.setup_distributed_logging(
        log_dir=str(tmp_path),
        log_level='INFO',
        component='instance-manager',
        identity='im-command-context',
    )
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    executor = LoggingExecutor()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='im-command-context',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        executor=executor,
        driver_factories={'nexus_live': lambda **_: BlockingDriver()},
    )
    await service.start()
    try:
        async with session_factory() as session:
            execution_repo = ExecutionRepository(session, shard_count=8)
            execution = await execution_repo.create_execution(
                CreateExecution(
                    execution_id='live-command-log-context',
                    mode=ExecutionMode.LIVE,
                    spec_type='nexus_live',
                    instance_mode='asyncio',
                    spec={'task_name': 'identity'},
                    route_status=RouteStatus.ACTIVE,
                ),
                now=utcnow(),
            )
            scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
            assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
            await session.commit()

        assert assignment is not None
        await service.poll_assignments_once()
        await asyncio.wait_for(executor.started.wait(), timeout=2)

        async with session_factory() as session:
            manager = ManagerRepository(session)
            await manager.mark_running(
                assignment.assignment_id,
                assignment.manager_id,
                assignment.manager_session_id,
                endpoint_url='http://127.0.0.1:9001/v1/internal/executions/live-command-log-context',
                endpoint_ready=True,
                runtime_info={},
                now=utcnow(),
            )
            await manager.mark_waiting_input(
                assignment.assignment_id,
                assignment.manager_id,
                assignment.manager_session_id,
                payload={'prompt': 'choose'},
                now=utcnow(),
            )
            gateway = GatewayRepository(session, manager_healthy_ttl=timedelta(seconds=15))
            command = await gateway.submit_command(
                'live-command-log-context',
                command_type='input.submit',
                payload={'text': 'Alpha'},
                idempotency_key='logging-command',
                now=utcnow(),
            )
            await session.commit()

        assert await service.claim_commands_once() == 1
        await asyncio.wait_for(executor.submitted.wait(), timeout=2)
        _flush_root_handlers()

        payloads = _json_log_lines(tmp_path / 'instance-manager-im-command-context.jsonl')
        payload = next(item for item in payloads if item['message'] == 'executor command log')
        assert payload['execution_id'] == 'live-command-log-context'
        assert payload['assignment_id'] == assignment.assignment_id
        assert payload['attempt_id'] == assignment.attempt_id
        assert payload['command_id'] == command.command_id
        assert payload['manager_id'] == 'im-command-context'
        assert payload['manager_session_id'] == service.session_id
    finally:
        await service.stop()


async def test_instance_manager_uses_execution_stop_timeout_for_stop_request(session_factory):
    config = DistributedConfig(shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
    driver = BlockingDriver()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='im-1',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={'nexus_live': lambda **_: driver},
    )
    await service.start()
    captured = {}

    async def fake_stop_assignment(assignment_id, timeout=10.0):
        captured['assignment_id'] = assignment_id
        captured['timeout'] = timeout

    service.executor.stop_assignment = fake_stop_assignment
    try:
        async with session_factory() as session:
            execution_repo = ExecutionRepository(session, shard_count=8)
            execution = await execution_repo.create_execution(
                CreateExecution(
                    execution_id='live-stop-timeout',
                    mode=ExecutionMode.LIVE,
                    spec_type='nexus_live',
                    instance_mode='asyncio',
                    spec={'task_name': 'identity'},
                    route_status=RouteStatus.ACTIVE,
                    stop_timeout_seconds=3,
                ),
                now=utcnow(),
            )
            scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
            assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
            await session.commit()

        assert assignment is not None
        await service.poll_assignments_once()
        await asyncio.wait_for(driver.started.wait(), timeout=2)

        async with session_factory() as session:
            repo = ExecutionRepository(session, shard_count=8)
            await repo.request_stop('live-stop-timeout', now=utcnow())
            await session.commit()

        assert await service.claim_commands_once() == 1
        assert captured == {'assignment_id': assignment.assignment_id, 'timeout': 3.0}
    finally:
        await service.stop()


async def test_instance_manager_reconciles_timed_out_stop_that_removed_assignment(session_factory) -> None:
    class TimedOutStopExecutor(LoggingExecutor):
        async def stop_assignment(self, assignment_id, timeout=10.0):  # noqa: ARG002
            self._assignments.pop(assignment_id, None)
            raise TimeoutError('executor stop bridge timed out')

    executor = TimedOutStopExecutor()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(aiomonitor_enabled=False, shard_count=8),
        manager_id='im-stop-timeout-removed',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        executor=executor,
        driver_factories={'nexus_live': lambda **_: BlockingDriver()},
    )
    await service.start()
    try:
        async with session_factory() as session:
            execution_repo = ExecutionRepository(session, shard_count=8)
            execution = await execution_repo.create_execution(
                CreateExecution(
                    execution_id='live-stop-timeout-removed',
                    mode=ExecutionMode.LIVE,
                    spec_type='nexus_live',
                    instance_mode='asyncio',
                    spec={'task_name': 'identity'},
                    route_status=RouteStatus.ACTIVE,
                ),
                now=utcnow(),
            )
            assignment = await SchedulerRepository(
                session,
                shard_count=8,
                manager_healthy_ttl=timedelta(seconds=15),
            ).claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
            await session.commit()

        assert assignment is not None
        assert await service.poll_assignments_once() == 1
        async with session_factory() as session:
            await ExecutionRepository(session, shard_count=8).request_stop(execution.execution_id, now=utcnow())
            await session.commit()

        assert await service.claim_commands_once() == 1
        async with session_factory() as session:
            current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)
            assert current.status == ExecutionStatus.KILLED
            command = await session.scalar(
                select(ExecutionCommandRow).where(ExecutionCommandRow.execution_id == execution.execution_id)
            )
            assert command is not None
            assert command.status == CommandStatus.DONE
    finally:
        with contextlib.suppress(TimeoutError):
            await service.stop()


async def test_instance_manager_manager_loop_bridge_has_a_caller_deadline(session_factory) -> None:
    release = threading.Event()
    entered = threading.Event()
    loop_owner = AsyncioExecutor(bridge_timeout=1.0)
    await loop_owner.start()
    service = InstanceManagerService(
        session_factory=session_factory,
        config=DistributedConfig(aiomonitor_enabled=False),
        manager_id='im-manager-bridge-timeout',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        executor=LoggingExecutor(),
    )
    try:
        service.manager_loop = loop_owner._loop
        assert service.manager_loop is not None

        def block_loop() -> None:
            entered.set()
            release.wait()

        service.manager_loop.call_soon_threadsafe(block_loop)
        assert await asyncio.to_thread(entered.wait, 1)

        with pytest.raises(TimeoutError):
            await service.call_manager_loop(asyncio.sleep(0), timeout=0.02)
    finally:
        release.set()
        await loop_owner.close()
