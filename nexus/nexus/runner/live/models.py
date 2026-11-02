"""Pydantic models for the live runner HTTP API."""

from __future__ import annotations

import json
import re

from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, Field

from nexus.core.workflow import WorkflowState
from nexus.runner.single import ModuleObjectConfig, ProcessorConfig, RunnerConfig


RunStatus = Literal['starting', 'idle', 'running', 'cached', 'killing', 'closed', 'failed', 'killed']
_CACHE_KEY_SCHEMA_VERSION = 1
_PATH_SEGMENT_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_-]*$')


class LiveCacheConfig(BaseModel):
    """Explicit live run cache identity configuration."""

    namespace: str = ''
    key_paths: list[str] = Field(default_factory=list)
    enabled: bool = True


class LiveConfig(BaseModel):
    """Live-service observation settings, independent from the task type."""

    working_dir: str = '/workspace'
    cache: LiveCacheConfig | None = None


class CreateRunFromYamlRequest(BaseModel):
    """Create a live run from a normal RunInput YAML document."""

    run_input_yaml: str
    live: LiveConfig = Field(default_factory=LiveConfig)


class CreateRunRequest(BaseModel):
    """Create a long-lived interactive run."""

    run_id: str | None = None
    imports: list[str] = Field(default_factory=list)
    openai_config_path: str | None = None

    runtime_provider_name: str | None = None
    runtime_provider_config: dict[str, Any] = Field(default_factory=dict)
    runtime_configs: dict[str, dict[str, Any]] = Field(default_factory=dict)

    openai_clients: dict[str, dict[str, Any]] = Field(default_factory=dict)
    tool_configs: dict[str, Any] = Field(default_factory=dict)

    task_name: str
    task_config: dict[str, Any] = Field(default_factory=dict)
    agent_name: str | None = None
    agent_config: dict[str, Any] = Field(default_factory=dict)
    processors: dict[str, ProcessorConfig] = Field(default_factory=dict)
    gateway_trajectory_observers: dict[str, ModuleObjectConfig] = Field(default_factory=dict)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    workflow_states: dict[str, WorkflowState] = Field(default_factory=dict)

    live: LiveConfig = Field(default_factory=LiveConfig)


class CreateRunResponse(BaseModel):
    run_id: str
    status: RunStatus
    events_cursor: int
    reused_cache: bool = False
    cache_key: str | None = None


class SubmitTurnRequest(BaseModel):
    text: str


class SubmitTurnResponse(BaseModel):
    run_id: str
    turn_id: str
    status: RunStatus


class RuntimeInfo(BaseModel):
    name: str
    runtime_id: str | None = None
    base_url: str | None = None
    proxy_url: str | None = None
    debug_gui_url: str | None = None


class RuntimeListResponse(BaseModel):
    runtimes: dict[str, RuntimeInfo] = Field(default_factory=dict)


class RunInfo(BaseModel):
    run_id: str
    status: RunStatus
    events_cursor: int
    cache_key: str | None = None
    current_turn_id: str | None = None
    runtime_base_url: str | None = None
    runtimes: dict[str, RuntimeInfo] = Field(default_factory=dict)
    error: str | None = None
    output: dict[str, Any] | None = None


class RunSummary(BaseModel):
    run_id: str
    status: RunStatus
    events_cursor: int
    cache_key: str | None = None
    current_turn_id: str | None = None
    runtime_base_url: str | None = None
    error: str | None = None


class RunListResponse(BaseModel):
    runs: list[RunInfo] = Field(default_factory=list)


class RunSummaryListResponse(BaseModel):
    runs: list[RunSummary] = Field(default_factory=list)


class RunConfigResponse(BaseModel):
    run_id: str
    request: dict[str, Any] = Field(default_factory=dict)


def _validate_cache_key_path(path: str) -> list[str]:
    parts = path.split('.')
    if not path or any(not part for part in parts) or any(_PATH_SEGMENT_RE.fullmatch(part) is None for part in parts):
        raise ValueError(f'invalid live cache key path: {path}')
    return parts


def _resolve_cache_key_path(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in _validate_cache_key_path(path):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f'missing live cache key path: {path}')
        current = current[part]
    return current


def request_cache_key(request: CreateRunRequest | dict[str, Any]) -> str | None:
    if isinstance(request, CreateRunRequest):
        payload = request.model_dump(mode='json', exclude={'run_id'})
    else:
        payload = dict(request)
        payload.pop('run_id', None)
    live = payload.get('live')
    cache = live.get('cache') if isinstance(live, dict) else None
    if not isinstance(cache, dict) or cache.get('enabled') is False:
        return None
    key_paths = cache.get('key_paths') or []
    if not key_paths:
        return None
    selected = {path: _resolve_cache_key_path(payload, path) for path in key_paths}
    canonical = json.dumps(
        {
            'schema_version': _CACHE_KEY_SCHEMA_VERSION,
            'namespace': cache.get('namespace') or '',
            'selected': selected,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        default=str,
    )
    return f'sha256:{sha256(canonical.encode("utf-8")).hexdigest()}'
