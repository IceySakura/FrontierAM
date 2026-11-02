"""Cloud export types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast


class ExportError(RuntimeError):
    """A safe, user-facing export failure."""


class TraceDatabaseError(ExportError):
    """A Trace Database failure whose class decides whether a fallback is safe.

    Only transient subclasses may fall back to the Application API: retrying an
    authentication or schema failure as an HTTP request would hide a
    configuration error behind a slower, misleading failure.
    """


class TraceDatabaseTimeout(TraceDatabaseError):  # noqa: N818 - reads as a condition
    """The server did not finish a statement before the client deadline."""


class TraceDatabaseUnavailable(TraceDatabaseError):  # noqa: N818 - reads as a condition
    """The connection could not be established or was lost."""


class TraceDatabaseThrottled(TraceDatabaseError):  # noqa: N818 - reads as a condition
    """The server rejected the read because of load or a lock conflict."""


class TraceDatabaseInterrupted(TraceDatabaseError):  # noqa: N818 - reads as a condition
    """The server aborted the statement (shutdown, query timeout, kill)."""


class TraceDatabaseAuth(TraceDatabaseError):  # noqa: N818 - reads as a condition
    """Credentials, grants or database selection are wrong."""


class TraceDatabaseSchema(TraceDatabaseError):  # noqa: N818 - reads as a condition
    """The table or a required column does not exist."""


class TraceDataInvalid(ExportError):  # noqa: N818 - reads as a condition
    """The Trace Database returned a row the export cannot interpret."""


class TraceProcessingError(ExportError):
    """Client-side work failed after the database row was transferred."""


class TraceProcessingTimeout(TraceProcessingError):  # noqa: N818 - reads as a condition
    """One item exceeded the client-side processing deadline."""


class ExportDeadlineExceeded(ExportError):  # noqa: N818 - reads as a condition
    """The whole export exceeded its client-side deadline."""


ResultStatus = Literal['success', 'failed', 'all']


def resolve_result_status(*, allow_partial: bool, result_status: str | None) -> ResultStatus:
    """Apply the success default only to partial Batch exports."""
    if result_status is None:
        return 'success' if allow_partial else 'all'
    if not allow_partial:
        raise ExportError('--result-status requires --allow-partial')
    if result_status not in ('success', 'failed', 'all'):
        raise ExportError('--result-status must be success, failed or all')
    return cast(ResultStatus, result_status)


_STATUS_MAP = {
    'succeeded': 'success',
    'failed': 'failed',
    'lost': 'failed',
    'expired': 'timeout',
    'killed': 'cancelled',
    'cancelled': 'cancelled',
}


_JSONL_FIELDS = (
    'run_id',
    'status',
    'start_time',
    'end_time',
    'retry_count',
    'error',
    'task_output',
    'agent_output',
    'processor_results',
    'metadata',
)


_LOCAL_TERMINAL_STATUSES = frozenset(_STATUS_MAP.values())


@dataclass(frozen=True, slots=True)
class TraceResult:
    output: dict[str, Any] | None = None
    start_time: str | None = None
    end_time: str | None = None
