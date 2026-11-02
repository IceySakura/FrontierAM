"""Deterministic task and event contracts for the hy-live control plane."""

from __future__ import annotations

import time

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TaskStatus = Literal['queued', 'running', 'done', 'failed', 'cancelled']
OwnerEventKind = Literal['permission_request', 'backend_input_request', 'task_delivery']
TaskExecutionStatus = Literal['success', 'failed', 'timeout', 'cancelled']
TaskFailureKind = Literal['execution', 'infrastructure', 'timeout', 'cancel']


class TaskError(BaseModel):
    model_config = ConfigDict(extra='forbid')

    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class TaskResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    execution_status: TaskExecutionStatus = 'success'
    failure_kind: TaskFailureKind | None = None
    error: TaskError | None = None
    final_output: str = ''
    trajectory: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    partial: bool = False

    @model_validator(mode='after')
    def validate_execution_contract(self) -> TaskResult:
        if self.execution_status == 'success':
            if self.failure_kind is not None or self.error is not None:
                raise ValueError('Successful TaskResult must not contain failure_kind or error')
            return self
        if self.execution_status == 'failed':
            has_valid_failure_kind = self.failure_kind in ('execution', 'infrastructure')
        elif self.execution_status == 'timeout':
            has_valid_failure_kind = self.failure_kind == 'timeout'
        else:
            has_valid_failure_kind = self.failure_kind == 'cancel'
        if not has_valid_failure_kind:
            raise ValueError(
                f'TaskResult execution_status {self.execution_status!r} has incompatible '
                f'failure_kind {self.failure_kind!r}'
            )
        if self.error is None:
            raise ValueError(f'TaskResult execution_status {self.execution_status!r} requires structured error')
        return self


class AgentTask(BaseModel):
    """One accepted foreground delegation; deliberately distinct from a Session."""

    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    task_id: str
    owner_id: str
    objective: str
    status: TaskStatus = 'queued'
    created_sequence: int
    coordinator_session_id: str
    target_session_id: str | None = None
    result: TaskResult | None = None
    error: TaskError | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    activity: list[dict[str, Any]] = Field(default_factory=list)
    """The latest harness Events of the task's newest Run as ``{id, kind, status, detail}``, newest last."""
    agent_message: str | None = None
    """Text of the newest harness Event of that Run."""


class SessionRef(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    session_id: str
    owner_id: str
    working_dir: str = '/workspace'


class OwnerEvent(BaseModel):
    model_config = ConfigDict(extra='forbid')

    sequence: int
    owner_id: str
    kind: OwnerEventKind
    task_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class ProjectCompletion(BaseModel):
    model_config = ConfigDict(extra='forbid')

    owner_id: str
    task_id: str
    session_id: str
    result: TaskResult


__all__ = [
    'AgentTask',
    'OwnerEvent',
    'ProjectCompletion',
    'SessionRef',
    'TaskError',
    'TaskExecutionStatus',
    'TaskFailureKind',
    'TaskResult',
    'TaskStatus',
]
