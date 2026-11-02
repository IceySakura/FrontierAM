from __future__ import annotations

import hashlib
import json

from collections.abc import Mapping
from contextlib import suppress
from typing import Any, cast, get_args

from nexus.distributed.execution.models import ExecutionStatus, RouteStatus
from nexus.distributed.execution.runtime_info import runtime_entries
from nexus.runner.live.models import (
    CreateRunRequest,
    RunInfo,
    RunStatus,
    RunSummary,
    RuntimeInfo,
    request_cache_key,
)


_VALID_RUN_STATUSES: set[str] = set(get_args(RunStatus))


def run_input_payload(payload: CreateRunRequest) -> dict[str, Any]:
    return {
        'run_id': payload.run_id,
        'imports': list(payload.imports),
        'openai_config_path': payload.openai_config_path,
        'runtime_provider_name': payload.runtime_provider_name,
        'runtime_provider_config': dict(payload.runtime_provider_config),
        'runtime_configs': dict(payload.runtime_configs),
        'openai_clients': dict(payload.openai_clients),
        'tool_configs': dict(payload.tool_configs),
        'task_name': payload.task_name,
        'task_config': dict(payload.task_config),
        'agent_name': payload.agent_name,
        'agent_config': dict(payload.agent_config),
        'processors': {key: value.model_dump(mode='json') for key, value in payload.processors.items()},
        'runner': payload.runner.model_dump(mode='json'),
        'workflow_states': {key: value.model_dump(mode='json') for key, value in payload.workflow_states.items()},
    }


def create_submission_payload(
    payload: CreateRunRequest,
    *,
    placement: Mapping[str, Any],
    idempotency_key: str,
    clear_run_id: bool = False,
) -> dict[str, Any]:
    run_input = run_input_payload(payload)
    if clear_run_id:
        run_input['run_id'] = None
    return {
        'submission_version': 1,
        'idempotency_key': idempotency_key,
        'environment_id': placement['environment'],
        'pool': placement['pool'],
        'code': {'mode': 'config_only'},
        'run_input': run_input,
        'live': payload.live.model_dump(mode='json'),
    }


def cache_key(payload: CreateRunRequest) -> str:
    configured = request_cache_key(payload)
    if configured is not None:
        return configured
    document = payload.model_dump(mode='json', exclude={'run_id'})
    canonical = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        default=str,
    ).encode('utf-8')
    return f'sha256:{hashlib.sha256(canonical).hexdigest()}'


def status_from_execution(execution: Mapping[str, Any]) -> RunStatus:
    if execution.get('route_status') == RouteStatus.CACHED:
        return 'cached'
    status = execution.get('status')
    if status in {ExecutionStatus.QUEUED, ExecutionStatus.ASSIGNED, ExecutionStatus.STARTING}:
        return 'starting'
    if status == ExecutionStatus.RUNNING:
        return 'running'
    if status == ExecutionStatus.WAITING_INPUT:
        return 'idle'
    if status == ExecutionStatus.STOPPING:
        return 'killing'
    if status == ExecutionStatus.SUCCEEDED:
        return 'closed'
    if status == ExecutionStatus.KILLED:
        return 'killed'
    if status in {ExecutionStatus.FAILED, ExecutionStatus.LOST, ExecutionStatus.EXPIRED}:
        return 'failed'
    return 'starting'


def runtime_info_from_payload(
    public_runs_base_path: str,
    run_id: str,
    runtime_info: Mapping[str, Any] | None,
) -> dict[str, RuntimeInfo]:
    raw_runtimes = runtime_entries(dict(runtime_info) if runtime_info is not None else None)
    runtimes: dict[str, RuntimeInfo] = {}
    for name, raw_runtime in raw_runtimes.items():
        base_path = _runtime_base_path(public_runs_base_path, run_id, name)
        runtimes[name] = RuntimeInfo(
            name=str(raw_runtime.get('name') or name),
            runtime_id=raw_runtime.get('runtime_id'),
            base_url=raw_runtime.get('base_url'),
            proxy_url=_rewrite_runtime_url(public_runs_base_path, raw_runtime.get('proxy_url'), run_id, name)
            or base_path,
            debug_gui_url=_rewrite_runtime_url(public_runs_base_path, raw_runtime.get('debug_gui_url'), run_id, name)
            or f'{base_path}/debug/gui',
        )
    return runtimes


def runtime_info_from_execution(
    public_runs_base_path: str,
    execution: Mapping[str, Any],
) -> dict[str, RuntimeInfo]:
    run_id = str(execution['execution_id'])
    assignment = execution.get('current_assignment') or {}
    runtime_info = assignment.get('runtime_info') if isinstance(assignment, Mapping) else None
    return runtime_info_from_payload(public_runs_base_path, run_id, runtime_info)


def run_info_from_execution(
    public_runs_base_path: str,
    execution: Mapping[str, Any],
    active_info: Mapping[str, Any] | None = None,
) -> RunInfo:
    run_id = str(execution['execution_id'])
    active_runtime_info = active_info.get('runtime_info') if isinstance(active_info, Mapping) else None
    runtimes = (
        runtime_info_from_payload(public_runs_base_path, run_id, active_runtime_info)
        if isinstance(active_runtime_info, Mapping)
        else runtime_info_from_execution(public_runs_base_path, execution)
    )
    first_runtime = next(iter(runtimes.values()), None)
    error = execution.get('error_json')
    live_info = active_info.get('live') if isinstance(active_info, Mapping) else None
    return RunInfo(
        run_id=run_id,
        status=_run_status_from_info(execution, live_info),
        events_cursor=_events_cursor_from_info(live_info),
        cache_key=cache_key_from_execution(execution),
        current_turn_id=_current_turn_id_from_info(execution, live_info),
        runtime_base_url=first_runtime.base_url if first_runtime else None,
        runtimes=runtimes,
        error=json.dumps(error, ensure_ascii=False) if isinstance(error, dict) else error,
        output=execution.get('result_json'),
    )


def run_summary_from_execution(public_runs_base_path: str, execution: Mapping[str, Any]) -> RunSummary:
    info = run_info_from_execution(public_runs_base_path, execution)
    return RunSummary(
        run_id=info.run_id,
        status=info.status,
        events_cursor=info.events_cursor,
        cache_key=info.cache_key,
        current_turn_id=info.current_turn_id,
        runtime_base_url=info.runtime_base_url,
        error=info.error,
    )


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        with suppress(ValueError):
            return int(value)
    return None


def cache_key_from_execution(execution: Mapping[str, Any]) -> str | None:
    request = request_from_execution(execution)
    if not request:
        return None
    try:
        return request_cache_key(request)
    except Exception:
        return None


def request_from_execution(execution: Mapping[str, Any]) -> dict[str, Any]:
    spec = execution.get('spec') or {}
    run_input = spec.get('run_input') if isinstance(spec, Mapping) else None
    request = dict(run_input) if isinstance(run_input, Mapping) else dict(spec)
    live = spec.get('live') if isinstance(spec, Mapping) else None
    if isinstance(live, Mapping):
        request['live'] = dict(live)
    request.setdefault('run_id', None)
    return request


def filter_executions(
    executions: list[dict[str, Any]],
    *,
    statuses: list[str] | None,
    cache_key: str | None,
    text_query: str | None,
) -> list[dict[str, Any]]:
    result = executions
    if statuses:
        status_set = set(statuses)
        result = [item for item in result if status_from_execution(item) in status_set]
    if cache_key:
        result = [item for item in result if cache_key_from_execution(item) == cache_key]
    if text_query:
        lowered = text_query.lower()
        result = [item for item in result if _execution_matches_query(item, lowered)]
    return result


def valid_run_statuses() -> frozenset[str]:
    return frozenset(_VALID_RUN_STATUSES)


def _runtime_base_path(public_runs_base_path: str, run_id: str, runtime_name: str) -> str:
    return f'{public_runs_base_path}/{run_id}/runtimes/{runtime_name}/proxy'


def _rewrite_runtime_url(
    public_runs_base_path: str,
    value: str | None,
    run_id: str,
    runtime_name: str,
) -> str | None:
    if not value:
        return value
    new_base = _runtime_base_path(public_runs_base_path, run_id, runtime_name)
    markers = (
        f'/v1/executions/{run_id}/runtimes/{runtime_name}/proxy',
        f'/v1/runs/{run_id}/runtimes/{runtime_name}/proxy',
    )
    for marker in markers:
        if value == marker or value.startswith(f'{marker}/'):
            return f'{new_base}{value[len(marker) :]}'
    return value


def _run_status_from_info(execution: Mapping[str, Any], live_info: Any) -> RunStatus:
    if isinstance(live_info, Mapping):
        status = live_info.get('status')
        if isinstance(status, str) and status in _VALID_RUN_STATUSES:
            return cast(RunStatus, status)
    return status_from_execution(execution)


def _events_cursor_from_info(live_info: Any) -> int:
    if isinstance(live_info, Mapping):
        cursor = int_or_none(live_info.get('events_cursor'))
        if cursor is not None:
            return cursor
    return 0


def _current_turn_id_from_info(execution: Mapping[str, Any], live_info: Any) -> str | None:
    if isinstance(live_info, Mapping) and live_info.get('current_turn_id') is not None:
        return str(live_info.get('current_turn_id'))
    assignment = execution.get('current_assignment') or {}
    runtime_info = assignment.get('runtime_info') if isinstance(assignment, Mapping) else None
    input_payload = runtime_info.get('input') if isinstance(runtime_info, Mapping) else None
    if isinstance(input_payload, Mapping) and input_payload.get('turn_id') is not None:
        return str(input_payload.get('turn_id'))
    return None


def _execution_matches_query(execution: Mapping[str, Any], lowered_query: str) -> bool:
    if lowered_query in str(execution.get('execution_id') or '').lower():
        return True
    request = request_from_execution(execution)
    for key in ('task_name', 'agent_name'):
        if lowered_query in str(request.get(key) or '').lower():
            return True
    resolved_cache_key = cache_key_from_execution(execution)
    return bool(resolved_cache_key and lowered_query in resolved_cache_key.lower())


__all__ = [
    'cache_key',
    'cache_key_from_execution',
    'create_submission_payload',
    'filter_executions',
    'int_or_none',
    'request_from_execution',
    'run_info_from_execution',
    'run_input_payload',
    'run_summary_from_execution',
    'runtime_info_from_execution',
    'runtime_info_from_payload',
    'status_from_execution',
    'valid_run_statuses',
]
