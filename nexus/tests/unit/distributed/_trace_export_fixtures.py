"""Shared builders for the Cloud trace export tests.

Underscore-prefixed so pytest does not collect it. The envelope shape and the
stop-after-resolution API stub were each rebuilt in every test that needed them, which made
a change to the published shape a change to four literals, and let two of them drift.
"""

from __future__ import annotations

from typing import Any

from nexus.distributed.export.types import ExportError
from nexus.runner.trace.tdsql.schema import principal_scope_id_from_table, principal_trace_table_name


PRINCIPAL_SCOPE_ID = '0123456789abcdef0123456789abcdef'
TABLE = principal_trace_table_name(PRINCIPAL_SCOPE_ID)
OTHER_PRINCIPAL_SCOPE_ID = 'b' * 32
PARTITION_KEY_ISO = '2026-09-21T10:30:00'
PARTITION_KEY_SQL = '2026-09-21 10:30:00.000000'


def principal_scope(table: str = TABLE, **extra: Any) -> dict[str, Any]:
    """Sink metadata for ``table``, as the principal writer publishes it."""
    return {
        'storage_scope': 'principal_v1',
        'principal_scope_id': principal_scope_id_from_table(table),
        **extra,
    }


def sink(name: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {'name': name, 'metadata': metadata}


def envelope_execution(
    *sinks: dict[str, Any],
    spec_type: str = 'nexus_single',
    execution_id: str = 'exe',
    **fields: Any,
) -> dict[str, Any]:
    """An Execution carrying ``sinks`` in the artifact envelope the writer publishes.

    ``live`` and ``single`` name the same structure under different keys; passing
    ``spec_type`` keeps that mapping in one place instead of at each call site.
    """
    key = 'live' if spec_type == 'nexus_live' else 'single'
    return {
        'execution_id': execution_id,
        'spec_type': spec_type,
        'runtime_info': {'_artifacts': {key: {'sinks': list(sinks)}}},
        **fields,
    }


def metadata_execution(metadata: dict[str, Any], *, spec_type: str = 'nexus_single', **fields: Any) -> dict[str, Any]:
    """An Execution with exactly one sink named ``my-db``."""
    return envelope_execution(sink('my-db', metadata), spec_type=spec_type, **fields)


def stub_api(execution: dict[str, Any], *, stop_at: str = 'get_batch_summary') -> Any:
    """Serve ``execution``, then fail at ``stop_at`` so a test can stop after resolution.

    The raised ``ExportError`` is the assertion boundary: resolution has happened by the
    time the next call is made, and nothing past it is under test.
    """

    def stop(name: str) -> None:
        if name == stop_at:
            raise ExportError('stop after resolution')

    class StubApi:
        async def get_execution(self, execution_id: str) -> dict[str, Any]:
            return execution

        async def get_batch_summary(self, execution_id: str) -> dict[str, Any]:
            stop('get_batch_summary')
            return {'total': 1, 'terminal': 1, 'runs': []}

        async def get_result(self, execution_id: str) -> dict[str, Any]:
            stop('get_result')
            return {}

        async def get_control_events(self, *args: Any, **kwargs: Any) -> list[Any]:
            stop('get_control_events')
            return []

    return StubApi()


def reader_stub(*, table: str = TABLE, trace_backend: str | None = None) -> Any:
    """Duck-typed trace reader, which is all the results exporter requires of one."""

    class Reader:
        pass

    reader = Reader()
    reader.table = table
    reader.parent_created_at = None
    if trace_backend is not None:
        reader.trace_backend = trace_backend
    return reader


__all__ = [
    'OTHER_PRINCIPAL_SCOPE_ID',
    'PARTITION_KEY_ISO',
    'PARTITION_KEY_SQL',
    'PRINCIPAL_SCOPE_ID',
    'TABLE',
    'envelope_execution',
    'metadata_execution',
    'principal_scope',
    'reader_stub',
    'sink',
    'stub_api',
]
