from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from nexus.core.agent import WithMainTrajectory
from nexus.core.task import TaskRunInput, TaskRunOutput, WithAgentRunner, WithScore

from .mode import BenchmarkServerMode


class BenchmarkDataset(str, Enum):
    TOOLATHLON = 'toolathlon'
    ATLAS = 'atlas'
    UNIVERSE = 'universe'


class BenchmarkCommand(BaseModel):
    command: str
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    timeout_s: float = 600.0


class BenchmarkEvaluator(BaseModel):
    type: str
    config: dict[str, Any] = Field(default_factory=dict)


class BenchmarkCleanup(BaseModel):
    server: str
    tool: str = ''
    cleanup_func: str
    cleanup_args: dict[str, Any] = Field(default_factory=dict)


class BenchmarkServiceConnection(BaseModel):
    placeholder: str
    internal_port: int
    value_template: str


class BenchmarkService(BaseModel):
    name: str
    runtime_name: str
    image: str
    exposed_ports: list[int] = Field(default_factory=list)
    working_dir: str | None = None
    runtime_config: dict[str, Any] = Field(default_factory=dict)
    startup: BenchmarkCommand | None = None
    ready_check: BenchmarkCommand | None = None
    ready_timeout_s: float = 120.0
    ready_poll_interval_s: float = 2.0
    connections: list[BenchmarkServiceConnection] = Field(default_factory=list)


class BenchmarkInputFile(BaseModel):
    path: str
    content: str
    encoding: Literal['utf8', 'base64'] = 'utf8'


class BenchmarkCase(BaseModel):
    dataset: BenchmarkDataset
    case_id: str
    server_mode: BenchmarkServerMode = BenchmarkServerMode.API
    prompt: str
    system_prompt: str | list[dict[str, Any]] | None = None
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    services: list[BenchmarkService] = Field(default_factory=list)
    enabled_tools: list[str] | None = None
    max_iterations: int = 30
    tool_output_truncation: int | None = 12000
    runtime_name: str = 'main'
    case_source_dir: str | None = None
    initial_workspace_dir: str | None = None
    input_files: list[BenchmarkInputFile] = Field(default_factory=list)
    case_text_replacements: dict[str, str] = Field(default_factory=dict)
    bootstraps: list[BenchmarkCommand] = Field(default_factory=list)
    preprocess: BenchmarkCommand | None = None
    evaluator: BenchmarkEvaluator
    cleanups: list[BenchmarkCleanup] = Field(default_factory=list)
    context_env: dict[str, str] = Field(default_factory=dict)
    output_format: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_task_config(self) -> dict[str, Any]:
        return {
            'dataset': self.dataset.value,
            'case_id': self.case_id,
            'server_mode': self.server_mode.value,
            'prompt': self.prompt,
            'system_prompt': self.system_prompt,
            'mcp_servers': self.mcp_servers,
            'services': [service.model_dump(mode='json') for service in self.services],
            'enabled_tools': self.enabled_tools,
            'max_iterations': self.max_iterations,
            'tool_output_truncation': self.tool_output_truncation,
            'runtime_name': self.runtime_name,
            'case_source_dir': self.case_source_dir,
            'initial_workspace_dir': self.initial_workspace_dir,
            'input_files': [item.model_dump(mode='json') for item in self.input_files],
            'case_text_replacements': self.case_text_replacements,
            'bootstraps': [command.model_dump(mode='json') for command in self.bootstraps],
            'preprocess': None if self.preprocess is None else self.preprocess.model_dump(mode='json'),
            'evaluator': self.evaluator.model_dump(mode='json'),
            'cleanups': [cleanup.model_dump(mode='json') for cleanup in self.cleanups],
            'context_env': self.context_env,
            'output_format': self.output_format,
            'metadata': self.metadata,
        }


class MCPBenchmarkTaskInput(TaskRunInput, WithAgentRunner):
    dataset: BenchmarkDataset
    case_id: str
    server_mode: BenchmarkServerMode = BenchmarkServerMode.API
    prompt: str
    system_prompt: str | list[dict[str, Any]] | None = None
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    services: list[BenchmarkService] = Field(default_factory=list)
    enabled_tools: list[str] | None = None
    max_iterations: int = 30
    tool_output_truncation: int | None = 12000
    runtime_name: str = 'main'
    case_source_dir: str | None = None
    initial_workspace_dir: str | None = None
    input_files: list[BenchmarkInputFile] = Field(default_factory=list)
    case_text_replacements: dict[str, str] = Field(default_factory=dict)
    bootstraps: list[BenchmarkCommand] = Field(default_factory=list)
    preprocess: BenchmarkCommand | None = None
    evaluator: BenchmarkEvaluator
    cleanups: list[BenchmarkCleanup] = Field(default_factory=list)
    context_env: dict[str, str] = Field(default_factory=dict)
    output_format: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MCPBenchmarkTaskOutput(TaskRunOutput, WithScore, WithMainTrajectory):
    dataset: BenchmarkDataset
    case_id: str
    server_mode: BenchmarkServerMode = BenchmarkServerMode.API
    passed: bool
    eval_details: dict[str, Any] = Field(default_factory=dict)
    agent_final_answer: str = ''
    loaded_tools: list[str] = Field(default_factory=list)
    agent_workspace: str | None = None


class EvalResult(BaseModel):
    passed: bool
    score: float
    details: dict[str, Any] = Field(default_factory=dict)
