"""Batch runner configuration and data models."""

from __future__ import annotations

import re

from typing import Any

from pydantic import BaseModel, Field

from nexus.runner.single import RunInput


_RUN_ID_UNSAFE = re.compile(r'[^a-zA-Z0-9_-]')


def sanitize_run_id(run_id: str) -> str:
    """Normalize a run ID into the character set used for database keys and directory names.

    The mapping is lossy: any character outside ``[a-zA-Z0-9_-]`` becomes ``_``, so distinct
    inputs can collapse into one run ID. Producers of run IDs must check for collisions.
    """
    return _RUN_ID_UNSAFE.sub('_', run_id)


class BatchItem(BaseModel):
    """Single batch task item with config and metadata.

    Attributes:
        config: RunInput configuration for this task
        run_id: Custom run identifier (directory name). If None, auto-generated as 00000, 00001, etc.
        metadata: Arbitrary metadata stored in run_dir/metadata.json. Useful for grouping, tagging, etc.
    """

    config: RunInput
    run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchConfig(BaseModel):
    """Batch execution configuration.

    Attributes:
        max_concurrent: Maximum number of concurrent tasks (default: 1)
        fail_fast: Stop all tasks immediately on first failure (default: False)
        retry_failed: Number of retries for failed tasks (default: 0)
        resume_workflow_on_retry: Resume committed workflow nodes on immediate retries (default: False)
        timeout_per_run: Timeout in seconds for each task (default: None = no timeout)
        resume: Skip already succeeded tasks (default: False)
        retry_only: Only retry previously failed tasks (default: False)
        enable_trace_db: Enable trace events → database (default: True)
        enable_log_capture: Enable log capture → database (default: True)
        log_dual_write: Dual-write logs to console + database (default: True)
        log_chunk_size_mb: Max log chunk size in MB (default: 4.0)
        log_chunk_time_seconds: Max time before log chunk flush in seconds (default: 60.0)
        enable_conflict_groups: Enable mutex groups from metadata.conflict_groups (default: False).
            When enabled, tasks sharing the same conflict group run serially while
            tasks in different groups can run concurrently.
        worker_processes: Number of local worker processes used by run-batch (default: 1).
            Values greater than 1 split the batch into per-worker shard databases.
        use_gateway: Wrap each run's main client as gateway and inject gateway_trajectory.
    """

    max_concurrent: int = 1
    worker_processes: int = 1
    fail_fast: bool = False
    retry_failed: int = 0
    resume_workflow_on_retry: bool = False
    timeout_per_run: int | None = None
    resume: bool = False
    retry_only: bool = False
    enable_trace_db: bool = True
    enable_log_capture: bool = True
    log_dual_write: bool = True
    log_chunk_size_mb: float = 4.0
    log_chunk_time_seconds: float = 60.0
    enable_conflict_groups: bool = False
    use_gateway: bool = False


class RunStatus(BaseModel):
    """Status of a single run."""

    run_id: str
    status: str  # pending|running|success|failed|timeout|cancelled
    start_time: str | None = None
    end_time: str | None = None
    error: str | None = None
    retry_count: int = 0


class BatchSummary(BaseModel):
    """Summary of batch execution results."""

    total: int
    pending: int = 0
    running: int = 0
    success: int = 0
    failed: int = 0
    timeout: int = 0
    cancelled: int = 0
    dry_run: bool = False
    failed_runs: list[str] = Field(default_factory=list)
    duration_seconds: float | None = None
    start_time: str | None = None
    end_time: str | None = None


__all__ = [
    'BatchItem',
    'BatchConfig',
    'RunStatus',
    'BatchSummary',
]
