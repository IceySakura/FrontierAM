"""Compatible public entry points for Cloud exports."""

from nexus.distributed.export.cli import (
    create_trace_reader,
    main,
    parse_args,
    resolve_secret,
    run_cloud_export,
    run_trace_export,
)
from nexus.distributed.export.client import CloudApi
from nexus.distributed.export.follow import export_trace_events
from nexus.distributed.export.results import (
    TdsqlTraceReader,
    batch_metadata,
    batch_snapshot,
    execution_snapshot,
    export_execution,
    extract_run_output,
    make_jsonl_record,
    normalize_status,
    project_lifecycle_times,
    validate_batch_summary,
    validate_execution,
    validate_trace_table,
)
from nexus.distributed.export.types import ExportError, TraceResult
from nexus.distributed.export.writer import AtomicJsonOutput, AtomicJsonlOutput, JsonlEventWriter
from nexus.distributed.trace.models import TraceStreamKey


__all__ = [
    'AtomicJsonOutput',
    'AtomicJsonlOutput',
    'CloudApi',
    'ExportError',
    'JsonlEventWriter',
    'TdsqlTraceReader',
    'TraceResult',
    'TraceStreamKey',
    'batch_metadata',
    'batch_snapshot',
    'create_trace_reader',
    'execution_snapshot',
    'export_execution',
    'export_trace_events',
    'extract_run_output',
    'main',
    'make_jsonl_record',
    'normalize_status',
    'parse_args',
    'project_lifecycle_times',
    'resolve_secret',
    'run_cloud_export',
    'run_trace_export',
    'validate_batch_summary',
    'validate_execution',
    'validate_trace_table',
]
