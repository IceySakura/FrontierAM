from __future__ import annotations

from datetime import timedelta

import pytest

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.execution.models import ExecutionMode
from nexus.distributed.persistence.orm import ExecutionCredentialRow, create_schema
from nexus.distributed.repositories.execution import CreateExecution, ExecutionRepository
from nexus.distributed.repositories.execution_credentials import ExecutionCredentialRepository
from nexus.distributed.repositories.reconciler import ReconcilerRepository
from nexus.distributed.utils.clock import utcnow


@pytest.fixture
async def credential_session_factory(tmp_path):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "execution-credentials.db"}')
    await create_schema(engine)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def _create_execution_pair(session, *, now):
    executions = ExecutionRepository(session, shard_count=8)
    parent = await executions.create_execution(
        CreateExecution(
            execution_id='credential-parent',
            mode=ExecutionMode.JOB,
            spec_type='nexus_batch',
            instance_mode='asyncio',
            spec={'batch': {}},
        ),
        now=now,
    )
    child = await executions.create_execution(
        CreateExecution(
            execution_id='credential-child',
            mode=ExecutionMode.JOB,
            spec_type='nexus_single',
            instance_mode='asyncio',
            spec={'run_input': {'task_name': 'identity'}},
            parent_execution_id=parent.execution_id,
        ),
        now=now,
    )
    return parent, child


async def test_child_reads_parent_execution_credential_without_copying(credential_session_factory) -> None:
    now = utcnow()
    async with credential_session_factory() as session:
        parent, child = await _create_execution_pair(session, now=now)
        credentials = ExecutionCredentialRepository(session)
        await credentials.create_many(
            parent.execution_id,
            {'wedata': {'subject': 'alice', 'key': 'secret', 'id': 'epoch-1'}},
            expires_at=now + timedelta(minutes=5),
            now=now,
        )
        payload = await credentials.get_payload(child.execution_id, 'wedata', now=now)
        child_row = await session.get(ExecutionCredentialRow, (child.execution_id, 'wedata'))

    assert payload == {'subject': 'alice', 'key': 'secret', 'id': 'epoch-1'}
    assert child_row is None


async def test_terminal_parent_keeps_credentials_until_children_stop_then_reconciles(credential_session_factory):
    from nexus.distributed.persistence.orm import ExecutionRow

    now = utcnow()
    async with credential_session_factory() as session:
        parent, child = await _create_execution_pair(session, now=now)
        credentials = ExecutionCredentialRepository(session)
        await credentials.create_many(
            parent.execution_id, {'cos': {'secret_id': 'test', 'secret_key': 'secret'}}, expires_at=None, now=now
        )
        parent_row = await session.get(ExecutionRow, parent.execution_id)
        parent_row.status = 'failed'
        await session.flush()
        await ReconcilerRepository(session).cleanup_terminal(now=now)
        assert await credentials.get_payload(child.execution_id, 'cos', now=now) is not None
        child_row = await session.get(ExecutionRow, child.execution_id)
        child_row.status = 'killed'
        await session.flush()
        # A fresh reconciler must revisit a parent whose cleanup_done_at is already stamped.
        await ReconcilerRepository(session).cleanup_terminal(now=now + timedelta(seconds=1))
        assert await credentials.get_payload(child.execution_id, 'cos', now=now) is None
        await ReconcilerRepository(session).cleanup_terminal(now=now + timedelta(seconds=2))


async def test_expired_execution_credentials_are_hidden_and_cleaned_in_bounded_batches(
    credential_session_factory,
) -> None:
    now = utcnow()
    async with credential_session_factory() as session:
        parent, _ = await _create_execution_pair(session, now=now)
        credentials = ExecutionCredentialRepository(session)
        await credentials.create_many(
            parent.execution_id,
            {
                'wedata': {'subject': 'alice', 'key': 'expired', 'id': 'epoch-1'},
                'model': {'token': 'expired'},
            },
            expires_at=now - timedelta(seconds=1),
            now=now,
        )
        assert await credentials.get_payload(parent.execution_id, 'wedata', now=now) is None
        assert await credentials.list_kinds(parent.execution_id, now=now) == ()
        assert await credentials.delete_expired(now=now, limit=1) == 1
        assert await credentials.delete_expired(now=now, limit=1) == 1
        assert await credentials.delete_expired(now=now, limit=1) == 0


async def test_credentials_cannot_be_attached_directly_to_batch_children(credential_session_factory) -> None:
    now = utcnow()
    async with credential_session_factory() as session:
        _, child = await _create_execution_pair(session, now=now)
        with pytest.raises(ValueError, match='top-level execution'):
            await ExecutionCredentialRepository(session).create_many(
                child.execution_id,
                {'wedata': {'subject': 'alice', 'key': 'secret', 'id': 'epoch-1'}},
                expires_at=None,
                now=now,
            )
