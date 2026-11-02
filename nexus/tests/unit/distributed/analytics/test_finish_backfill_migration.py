from __future__ import annotations

import importlib

from datetime import timedelta

import pytest

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session
from tests.unit.distributed.analytics.test_statistics import NOW, execution, item

from nexus.distributed.persistence.db import _run_migrations
from nexus.distributed.persistence.migrations import HEAD_REVISION
from nexus.distributed.persistence.orm import Base, BatchItemRow


MODULE = 'nexus.distributed.persistence.migrations.versions.0034_backfill_batch_finished_at'


def legacy_database(tmp_path):
    engine = create_engine(f'sqlite:///{tmp_path}/finish-backfill.db')
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(
            [
                execution('batch', spec_type='nexus_batch'),
                execution('real-child', finished_at=NOW + timedelta(hours=1)),
                execution('wrong-state-child', status='failed', finished_at=NOW + timedelta(hours=2)),
            ]
        )
        records = [
            item('a-existing', finished_at=NOW, metadata_json={'keep': 1}),
            item('b-real', current_child_execution_id='real-child', finished_at=None),
            item('c-wrong-state', current_child_execution_id='wrong-state-child', finished_at=None),
            item('d-running', status='running', finished_at=None),
            item('e-pending', status='pending', finished_at=None),
        ]
        records += [
            item('f-' + status, status=status, finished_at=None)
            for status in ('succeeded', 'failed', 'lost', 'expired', 'killed', 'cancelled')
        ]
        for index, record in enumerate(records):
            record.task_ordinal = index
            record.updated_at = NOW + timedelta(days=3)
            record.status_version = 7
        session.add_all(records)
        session.commit()
    with engine.begin() as connection:
        Operations(MigrationContext.configure(connection)).drop_column(
            'distributed_batch_items', 'finished_at_estimated'
        )
        connection.exec_driver_sql('CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY)')
        connection.exec_driver_sql("INSERT INTO alembic_version VALUES ('0033_business_statistics')")
    return engine


def assert_recovered(engine):
    with Session(engine) as session:
        rows = {row.item_id: row for row in session.scalars(select(BatchItemRow))}
        assert rows['a-existing'].finished_at == NOW
        assert not rows['a-existing'].finished_at_estimated
        assert rows['a-existing'].metadata_json == {'keep': 1}
        assert rows['b-real'].finished_at == NOW + timedelta(hours=1)
        assert not rows['b-real'].finished_at_estimated
        for key, row in rows.items():
            assert row.updated_at == NOW + timedelta(days=3) and row.status_version == 7
            if key.startswith('f-') or key == 'c-wrong-state':
                assert row.finished_at == row.updated_at and row.finished_at_estimated
            if key in {'d-running', 'e-pending'}:
                assert row.finished_at is None and not row.finished_at_estimated


def test_upgrade_from_deployed_revision_recovers_real_times_then_marks_terminal_estimates(tmp_path, monkeypatch):
    module = importlib.import_module(MODULE)
    monkeypatch.setattr(module, 'PAGE_SIZE', 2)
    engine = legacy_database(tmp_path)
    try:
        with engine.connect() as connection:
            _run_migrations(connection, commit=True)
            assert connection.scalar(text('SELECT version_num FROM alembic_version')) == HEAD_REVISION
        assert_recovered(engine)
        with engine.begin() as connection:
            module.upgrade(Operations(MigrationContext.configure(connection)))
        assert_recovered(engine)
    finally:
        engine.dispose()


def test_interrupted_backfill_keeps_completed_pages_and_retries_without_restamping_them(tmp_path, monkeypatch):
    module = importlib.import_module(MODULE)
    monkeypatch.setattr(module, 'PAGE_SIZE', 2)
    engine = legacy_database(tmp_path)
    original = module.upgrade_batches

    def interrupted(operation):
        for page, value in enumerate(original(operation)):
            yield value
            if page == 1:
                raise RuntimeError('interrupted after committed page')

    try:
        with monkeypatch.context() as patch:
            patch.setattr(module, 'upgrade_batches', interrupted)
            with engine.connect() as connection, pytest.raises(RuntimeError, match='interrupted'):
                _run_migrations(connection, commit=True)
        with Session(engine) as session:
            assert session.scalar(text('SELECT version_num FROM alembic_version')) == '0033_business_statistics'
            assert session.get(BatchItemRow, ('batch', 'b-real')).finished_at == NOW + timedelta(hours=1)
            assert session.get(BatchItemRow, ('batch', 'c-wrong-state')).finished_at is None
        with engine.connect() as connection:
            _run_migrations(connection, commit=True)
        assert_recovered(engine)
    finally:
        engine.dispose()
