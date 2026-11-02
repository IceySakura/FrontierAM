"""Type definitions for the toolathlon_gym task extension."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from nexus.core.agent import WithMainTrajectory
from nexus.core.task import TaskRunInput, TaskRunOutput, WithAgentRunner, WithScore
from nexus.extensions.tasks.mcp_benchmark.mode import BenchmarkServerMode
from nexus.extensions.tasks.mcp_benchmark.types import BenchmarkCommand, BenchmarkEvaluator


class ToolathlonGymTaskInput(TaskRunInput, WithAgentRunner):
    """Input for a single Toolathlon-Gym RL training task."""

    case_id: str
    server_mode: BenchmarkServerMode = BenchmarkServerMode.MOCK
    prompt: str
    system_prompt: str | list[dict[str, Any]] | None = None
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    enabled_tools: list[str] | None = None
    max_iterations: int = 30
    observation_time_timeout: float | None = None
    tool_output_truncation: int | None = 12000
    runtime_name: str = 'main'
    case_source_dir: str | None = None
    local_case_source_dir: str | None = None
    local_credentials_dir: str | None = None
    initial_workspace_dir: str | None = None
    bootstraps: list[BenchmarkCommand] = Field(default_factory=list)
    preprocess: BenchmarkCommand | None = None
    evaluator: BenchmarkEvaluator
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Self-contained case data as base64-encoded tar.gz archives.
    # When set, these replace local_case_source_dir for uploading case files.
    case_data_tar_b64: str | None = None
    # Evaluation data injected AFTER agent finishes (anti reward-hacking).
    eval_tar_b64: str | None = None
    groundtruth_tar_b64: str | None = None
    # Immutable task/data contract embedded in self-contained parquet rows.
    # Bound rows select their isolation topology and fail closed on tampering.
    data_binding: dict[str, Any] | None = None


class ToolathlonGymTaskOutput(TaskRunOutput, WithScore, WithMainTrajectory):
    """Output from a single Toolathlon-Gym task execution."""

    case_id: str
    server_mode: BenchmarkServerMode = BenchmarkServerMode.MOCK
    passed: bool
    eval_details: dict[str, Any] = Field(default_factory=dict)
    agent_final_answer: str = ''
    loaded_tools: list[str] = Field(default_factory=list)
    agent_workspace: str | None = None
