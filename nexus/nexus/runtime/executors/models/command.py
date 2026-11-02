"""Command executor request/response models.

Pure Pydantic — no subprocess dependencies.
"""

from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, Field

from nexus.runtime.executors.models.base import ExecutorRequest


class CommandRunStatus(str, Enum):
    Finished = 'Finished'
    Error = 'Error'
    TimeLimitExceeded = 'TimeLimitExceeded'


class RunCommandResponse(BaseModel):
    status: CommandRunStatus
    execution_time: float | None = None
    return_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None


class RunCommandRequest(ExecutorRequest[RunCommandResponse]):
    executor_name: ClassVar[str] = 'run_command'
    response_type: ClassVar[type[BaseModel]] = RunCommandResponse

    command: str = Field(..., examples=['echo 123'], description='the command to run')
    cwd: str | None = None
    timeout: float | None = Field(
        180, description='command timeout in seconds; None delegates the deadline to the caller'
    )
    timeout_control_id: str | None = Field(None, description='active-time timeout control identifier')
    stdin: str | None = Field(None, examples=[''], description='optional string to pass into stdin')
    extra_env: dict[str, str] = Field(default_factory=dict)
